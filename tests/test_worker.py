"""Run: python -m unittest discover -s tests -p test_worker.py -v"""
import concurrent.futures
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
import wave

from worker.server import APIError, Config, LICENSE, MODELS, Worker, make_server, validate_job

TOKEN = "test-only-" + "x" * 40


def fake_generator(job, stem):
    with wave.open(str(stem) + ".wav", "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * int(job["duration"] * 16000))
    return 16000


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.workers = []
        self.servers = []

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        for worker in reversed(self.workers):
            worker.close()
        self.directory.cleanup()

    def worker(self, generator=fake_generator, start=False, **limits):
        result = Worker(Config(TOKEN, self.path, accept_noncommercial=True, **limits), generator, start=start)
        self.workers.append(result)
        return result

    def serve(self, worker):
        server = make_server(worker, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return server.server_address[1]

    def request(self, port, method, path, payload=None, token=TOKEN, raw=None, headers=None):
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        request_headers = {"Content-Type": "application/json"}
        if token is not None:
            request_headers["Authorization"] = "Bearer " + token
        request_headers.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_config_requires_explicit_nc_and_secret(self):
        for overrides in ({"accept_noncommercial": False}, {"token": "short"},
                          {"token": "x" * 32 + "\n"}, {"max_output_bytes": 0},
                          {"max_output_bytes": 1024**3 + 1}, {"artifact_ttl_seconds": float("nan")},
                          {"artifact_ttl_seconds": -1}, {"artifact_ttl_seconds": True}):
            values = {"token": TOKEN, "output_dir": self.path, "accept_noncommercial": True}
            values.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Config(**values)
        with patch.dict("os.environ", {"ACCEPT_NONCOMMERCIAL_MODELS": "false"}):
            with self.assertRaises(ValueError):
                Config.from_env()

    def test_strict_validation(self):
        good = {"kind": "music", "prompt": "piano"}
        self.assertEqual(validate_job(good)["duration"], 5)
        self.assertEqual(validate_job(good)["seed"], 0)
        invalid = [[], None, {**good, "extra": 1}, {"prompt": "a"},
                   *[{**good, "kind": value} for value in ("bad", None, [], True)],
                   *[{**good, "prompt": value} for value in ("", " ", "a" * 2001, 123, "\ud800")],
                   *[{**good, "duration": value} for value in (True, "5", None, -1, 0.49, 10.01, float("nan"), float("inf"), 10**1000)],
                   *[{**good, "seed": value} for value in (True, -1, 2**32, 0.5, "0", None)]]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(APIError) as caught:
                validate_job(payload)
            self.assertEqual(caught.exception.status, 400)
        for duration in (0.5, 10):
            validate_job({**good, "duration": duration, "seed": 2**32 - 1})

    def test_http_contract_auth_and_body_limits(self):
        worker = self.worker()
        port = self.serve(worker)
        self.assertEqual(self.request(port, "GET", "/healthz", token=None), (200, {"status": "ok"}))
        for token in (None, "wrong", ""):
            self.assertEqual(self.request(port, "GET", "/models", token=token)[0], 401)
        status, models = self.request(port, "GET", "/models")
        self.assertEqual(status, 200)
        self.assertEqual({row["id"] for row in models["models"]}, set(MODELS.values()))
        self.assertTrue(all(row["license"] == LICENSE for row in models["models"]))
        self.assertEqual(self.request(port, "POST", "/jobs", {"kind": "sound", "prompt": "birds"})[0], 202)
        for body in (b"{", b'{"kind":"music","prompt":"a","duration":NaN}', b'{"kind":"music","kind":"sound","prompt":"a"}', b"\xff", b"[" * 1100):
            self.assertEqual(self.request(port, "POST", "/jobs", raw=body)[0], 400)
        self.assertEqual(self.request(port, "POST", "/jobs", raw=b"x" * 16385)[0], 413)
        self.assertEqual(self.request(port, "POST", "/jobs", raw=b"{}", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request(port, "GET", "/jobs/not-a-uuid")[0], 400)
        self.assertEqual(self.request(port, "GET", "/jobs/" + str(uuid4()))[0], 404)
        self.assertEqual(self.request(port, "GET", "/missing")[0], 404)

    def test_queue_limit_atomic_under_concurrent_submit(self):
        worker = self.worker()
        def submit(_):
            try:
                return worker.submit({"kind": "music", "prompt": "piano"})["status"]
            except APIError as error:
                return error.status
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(submit, range(20)))
        self.assertEqual(results.count("queued"), 4)
        self.assertEqual(results.count(503), 16)

    def test_one_running_four_queued_and_real_wav_metadata(self):
        started, release = threading.Event(), threading.Event()
        active, maximum = 0, 0
        lock = threading.Lock()
        def blocking(job, stem):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test timeout")
            try:
                return fake_generator(job, stem)
            finally:
                with lock:
                    active -= 1
        worker = self.worker(blocking, start=True)
        job_id = worker.submit({"kind": "music", "prompt": "piano", "duration": 0.5, "seed": 7})["id"]
        self.assertTrue(started.wait(timeout=3))
        try:
            self.assertEqual(worker.get_job(job_id)["status"], "running")
            for _ in range(4):
                worker.submit({"kind": "sound", "prompt": "rain", "duration": 0.5})
            with self.assertRaises(APIError) as caught:
                worker.submit({"kind": "music", "prompt": "full"})
            self.assertEqual(caught.exception.status, 503)
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with worker.condition:
                pending = worker.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending == 0:
                break
            time.sleep(0.01)
        self.assertEqual(pending, 0)
        self.assertEqual(maximum, 1)
        self.assertEqual(worker.get_job(job_id)["artifact_id"], job_id)
        artifact = worker.artifact(job_id)
        self.assertEqual(artifact["filename"], job_id + ".wav")
        self.assertEqual(artifact["sample_rate"], 16000)
        self.assertEqual(artifact["seed"], 7)
        self.assertEqual(artifact["license"], LICENSE)
        self.assertEqual(artifact["size_bytes"], (self.path / artifact["filename"]).stat().st_size)
        self.assertFalse(list(self.path.glob("*.tmp.wav")))
        port = self.serve(worker)
        self.assertEqual(self.request(port, "GET", "/artifacts/" + job_id), (200, artifact))

    def test_recovery_fails_running_resumes_queued(self):
        worker = self.worker()
        interrupted = worker.submit({"kind": "music", "prompt": "interrupted"})["id"]
        queued = worker.submit({"kind": "sound", "prompt": "resume", "duration": 0.5})["id"]
        worker._claim()
        (self.path / (interrupted + ".tmp.wav")).write_bytes(b"partial")
        worker.close()
        self.workers.remove(worker)
        worker = self.worker(start=True)
        self.assertEqual(worker.get_job(interrupted)["status"], "failed")
        deadline = time.monotonic() + 5
        while worker.get_job(queued)["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(worker.get_job(queued)["status"], "succeeded")
        self.assertFalse((self.path / (interrupted + ".tmp.wav")).exists())

    def test_generation_errors_sanitized_and_temp_removed(self):
        def broken(job, stem):
            Path(str(stem) + ".wav").write_bytes(b"partial")
            raise RuntimeError("private details " + TOKEN)
        worker = self.worker(broken)
        job_id = worker.submit({"kind": "music", "prompt": "a"})["id"]
        with self.assertLogs("audiocraft.worker", level="ERROR") as logs:
            worker._generate(worker._claim())
        self.assertNotIn(TOKEN, "\n".join(logs.output))
        self.assertIn("private details", "\n".join(logs.output))
        result = worker.get_job(job_id)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "generation failed")
        self.assertEqual(list(self.path.glob("*.wav")), [])
        with self.assertRaises(APIError):
            worker.artifact(job_id)

    def test_cleanup_error_does_not_leave_job_running(self):
        def broken(job, stem):
            Path(str(stem) + ".wav").write_bytes(b"partial")
            raise RuntimeError("generation failure")
        worker = self.worker(broken)
        job_id = worker.submit({"kind": "music", "prompt": "a"})["id"]
        with patch.object(Path, "unlink", side_effect=PermissionError("test cleanup failure")):
            with self.assertLogs("audiocraft.worker", level="ERROR"):
                worker._generate(worker._claim())
        self.assertEqual(worker.get_job(job_id)["status"], "failed")

    def test_artifact_paths_reject_traversal(self):
        worker = self.worker()
        outside = self.path / "unowned.wav"
        outside.write_bytes(b"do not delete")
        for value in ("../unowned", "..%2funowned", str(uuid4()) + "/../unowned", "/etc/passwd", ""):
            with self.subTest(value=value), self.assertRaises(APIError) as caught:
                worker.artifact(value)
            self.assertEqual(caught.exception.status, 400)
        worker.prune()
        self.assertEqual(outside.read_bytes(), b"do not delete")

    def test_quota_and_ttl_only_prune_owned_finished_artifacts(self):
        worker = self.worker(max_output_bytes=20000, artifact_ttl_seconds=60)
        unrelated = self.path / (str(uuid4()) + ".wav")
        unrelated.write_bytes(b"untouched")
        ids = []
        for _ in range(2):
            job_id = worker.submit({"kind": "sound", "prompt": "rain", "duration": 0.5})["id"]
            worker._generate(worker._claim())
            ids.append(job_id)
        with self.assertRaises(APIError):
            worker.artifact(ids[0])
        self.assertIsNone(worker.get_job(ids[0])["artifact_id"])
        self.assertEqual(worker.artifact(ids[1])["id"], ids[1])
        with worker.condition, worker.db:
            worker.db.execute("UPDATE jobs SET updated_at=? WHERE id=?", (time.time() - 61, ids[1]))
        worker.prune()
        self.assertFalse((self.path / (ids[1] + ".wav")).exists())
        self.assertEqual(unrelated.read_bytes(), b"untouched")

    def test_oversized_generation_preserves_existing_artifact(self):
        worker = self.worker(max_output_bytes=20000)
        retained = worker.submit({"kind": "music", "prompt": "retain", "duration": 0.5})["id"]
        worker._generate(worker._claim())
        metadata = worker.artifact(retained)
        original = (self.path / metadata["filename"]).read_bytes()
        oversized = worker.submit({"kind": "sound", "prompt": "too large", "duration": 1})["id"]
        with self.assertLogs("audiocraft.worker", level="ERROR"):
            worker._generate(worker._claim())
        self.assertEqual(worker.get_job(oversized)["status"], "failed")
        self.assertEqual(worker.artifact(retained), metadata)
        self.assertEqual((self.path / metadata["filename"]).read_bytes(), original)
        port = self.serve(worker)
        self.assertEqual(self.request(port, "GET", "/artifacts/" + retained), (200, metadata))

    def test_single_artifact_larger_than_quota_fails(self):
        worker = self.worker(max_output_bytes=10)
        job_id = worker.submit({"kind": "music", "prompt": "a"})["id"]
        with self.assertLogs("audiocraft.worker", level="ERROR"):
            worker._generate(worker._claim())
        self.assertEqual(worker.get_job(job_id)["status"], "failed")
        self.assertFalse(list(self.path.glob("*.wav")))


if __name__ == "__main__":
    unittest.main()
