"""Private, serialized AudioCraft job worker. Model weights require NC opt-in."""
from __future__ import annotations

import gc
import hmac
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from uuid import UUID, uuid4

LOGGER = logging.getLogger("audiocraft.worker")
LICENSE = "CC-BY-NC-4.0"
MODELS = {"music": "facebook/musicgen-small", "sound": "facebook/audiogen-medium"}
MAX_BODY = 16 * 1024
JOB_FIELDS = ("id", "status", "kind", "prompt", "duration", "seed", "created_at", "updated_at", "error", "artifact_id")


class APIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Config:
    token: str
    output_dir: Path = Path("/data")
    max_output_bytes: int = 1024**3
    artifact_ttl_seconds: float = 86400
    accept_noncommercial: bool = False

    def __post_init__(self):
        if self.accept_noncommercial is not True:
            raise ValueError("ACCEPT_NONCOMMERCIAL_MODELS=true is required")
        if not isinstance(self.token, str) or len(self.token) < 32 or any(not 33 <= ord(c) <= 126 for c in self.token):
            raise ValueError("worker token must contain at least 32 non-whitespace ASCII characters")
        if type(self.max_output_bytes) is not int or not 1 <= self.max_output_bytes <= 1024**3:
            raise ValueError("MAX_OUTPUT_BYTES must be an integer between 1 and 1073741824")
        if type(self.artifact_ttl_seconds) not in (int, float) or not math.isfinite(self.artifact_ttl_seconds) or self.artifact_ttl_seconds <= 0:
            raise ValueError("ARTIFACT_TTL_SECONDS must be finite and positive")

    @classmethod
    def from_env(cls) -> Config:
        if os.environ.get("ACCEPT_NONCOMMERCIAL_MODELS") != "true":
            raise ValueError("ACCEPT_NONCOMMERCIAL_MODELS=true is required")
        token = Path(os.environ.get("WORKER_TOKEN_FILE", "/run/secrets/worker_token")).read_text(encoding="utf-8").strip()
        return cls(token, Path(os.environ.get("OUTPUT_DIR", "/data")),
                   int(os.environ.get("MAX_OUTPUT_BYTES", str(1024**3))),
                   float(os.environ.get("ARTIFACT_TTL_SECONDS", "86400")), True)


def validate_job(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"kind", "prompt", "duration", "seed"}:
        raise APIError(400, "expected job object with only kind, prompt, duration, seed")
    kind, prompt = payload.get("kind"), payload.get("prompt")
    duration, seed = payload.get("duration", 5), payload.get("seed", 0)
    if not isinstance(kind, str) or kind not in MODELS:
        raise APIError(400, "kind must be music or sound")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 2000 or not prompt.strip():
        raise APIError(400, "prompt must contain 1..2000 characters")
    try:
        prompt.encode("utf-8")
    except UnicodeError:
        raise APIError(400, "prompt must be valid Unicode") from None
    if type(duration) not in (int, float) or not 0.5 <= duration <= 10 or not math.isfinite(duration):
        raise APIError(400, "duration must be a finite number between 0.5 and 10")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise APIError(400, "seed must be an integer between 0 and 4294967295")
    return {"kind": kind, "prompt": prompt, "duration": float(duration), "seed": seed}


def validate_id(value: str) -> str:
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise APIError(400, "invalid UUID") from None
    return value


class AudioCraftGenerator:
    """Only the daemon thread touches models; imports remain lazy."""

    def __init__(self):
        self.model = None
        self.kind = None

    def __call__(self, job: dict, stem: Path) -> int:
        import torch
        from audiocraft.data.audio import audio_write
        from audiocraft.models import AudioGen, MusicGen

        if self.kind != job["kind"]:
            self.model = None
            self.kind = None
            gc.collect()
            torch.cuda.empty_cache()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required")
            model_class = MusicGen if job["kind"] == "music" else AudioGen
            self.model = model_class.get_pretrained(MODELS[job["kind"]], device="cuda")
            self.kind = job["kind"]
        torch.manual_seed(job["seed"])
        torch.cuda.manual_seed_all(job["seed"])
        self.model.set_generation_params(duration=job["duration"])
        with torch.inference_mode():
            waveform = self.model.generate([job["prompt"]], progress=False)[0].cpu()
        audio_write(str(stem), waveform, self.model.sample_rate, format="wav", strategy="loudness", loudness_compressor=True)
        return self.model.sample_rate


class Worker:
    def __init__(self, config: Config, generator: Callable[[dict, Path], int] | None = None, *, start: bool = True):
        self.config = config
        self.output_dir = config.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator if generator is not None else AudioCraftGenerator()
        self.condition = threading.Condition(threading.RLock())
        self.stopping = False
        self.thread = None
        self.db = sqlite3.connect(self.output_dir / "jobs.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, kind TEXT NOT NULL,
                prompt TEXT NOT NULL, duration REAL NOT NULL, seed INTEGER NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL, error TEXT,
                artifact_id TEXT, artifact TEXT)""")
            interrupted = self.db.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
            self.db.execute("UPDATE jobs SET status='failed', error='worker restarted during generation', updated_at=? WHERE status='running'", (time.time(),))
        for row in interrupted:
            self._path(row["id"]).unlink(missing_ok=True)
        for row in self.db.execute("SELECT id FROM jobs WHERE status IN ('failed','succeeded')"):
            (self.output_dir / (validate_id(row["id"]) + ".tmp.wav")).unlink(missing_ok=True)
        self.prune()
        if start:
            self.thread = threading.Thread(target=self._run, name="generation", daemon=True)
            self.thread.start()

    def log_exception(self, context: str):
        LOGGER.error("%s\n%s", context, traceback.format_exc().replace(self.config.token, "[REDACTED]"))

    def submit(self, payload: object) -> dict:
        job = validate_job(payload)
        now, job_id = time.time(), str(uuid4())
        with self.condition, self.db:
            if self.stopping or self.db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0] >= 4:
                raise APIError(503, "generation queue is full or unavailable")
            self.db.execute("INSERT INTO jobs (id,status,kind,prompt,duration,seed,created_at,updated_at) VALUES (?,'queued',?,?,?,?,?,?)",
                            (job_id, job["kind"], job["prompt"], job["duration"], job["seed"], now, now))
            self.condition.notify()
        return {"id": job_id, "status": "queued"}

    def get_job(self, job_id: str) -> dict:
        validate_id(job_id)
        with self.condition:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise APIError(404, "job not found")
            return {key: row[key] for key in JOB_FIELDS}

    def _path(self, job_id: str) -> Path:
        return self.output_dir / (validate_id(job_id) + ".wav")

    def artifact(self, job_id: str) -> dict:
        validate_id(job_id)
        with self.condition:
            self.prune()
            row = self.db.execute("SELECT artifact FROM jobs WHERE id=? AND status='succeeded'", (job_id,)).fetchone()
            path = self._path(job_id)
            if row is None or row["artifact"] is None or path.is_symlink() or not path.is_file():
                raise APIError(404, "artifact not found or expired")
            return json.loads(row["artifact"])

    def prune(self, incoming_bytes: int = 0):
        """Delete only artifacts owned by finished DB rows, never arbitrary paths."""
        if incoming_bytes > self.config.max_output_bytes:
            raise RuntimeError("artifact exceeds storage limit")
        with self.condition, self.db:
            rows = self.db.execute("SELECT id, updated_at FROM jobs WHERE status IN ('succeeded','failed') AND artifact_id IS NOT NULL ORDER BY updated_at, id").fetchall()
            files = []
            for row in rows:
                path = self._path(row["id"])
                size = path.stat().st_size if not path.is_symlink() and path.is_file() else 0
                files.append((row, path, size))
            total = sum(size for _, _, size in files) + incoming_bytes
            cutoff = time.time() - self.config.artifact_ttl_seconds
            for row, path, size in files:
                if row["updated_at"] <= cutoff or total > self.config.max_output_bytes or path.is_symlink() or not path.exists():
                    path.unlink(missing_ok=True)
                    self.db.execute("UPDATE jobs SET artifact_id=NULL, artifact=NULL WHERE id=?", (row["id"],))
                    total -= size
            if total > self.config.max_output_bytes:
                raise RuntimeError("artifact exceeds storage limit")

    def _claim(self) -> dict | None:
        with self.condition, self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, id LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE jobs SET status='running', updated_at=? WHERE id=?", (time.time(), row["id"]))
            return dict(row)

    def _generate(self, job: dict):
        job_id = validate_id(job["id"])
        # audio_write appends .wav; rename within the output filesystem is atomic.
        stem = self.output_dir / (job_id + ".tmp")
        temporary = stem.with_suffix(".tmp.wav")
        destination = self._path(job_id)
        try:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            sample_rate = self.generator(job, stem)
            if temporary.is_symlink() or not temporary.is_file():
                raise RuntimeError("generator did not produce a regular WAV")
            size = temporary.stat().st_size
            if type(sample_rate) is not int or sample_rate <= 0 or size <= 0:
                raise RuntimeError("generator returned invalid audio metadata")
            metadata = {"id": job_id, "filename": job_id + ".wav", "media_type": "audio/wav",
                        "size_bytes": size, "duration": job["duration"], "sample_rate": sample_rate,
                        "model": MODELS[job["kind"]], "seed": job["seed"], "license": LICENSE}
            with self.condition, self.db:
                self.prune(size)
                if destination.exists() or destination.is_symlink():
                    raise RuntimeError("artifact destination already exists")
                temporary.replace(destination)
                self.db.execute("UPDATE jobs SET status='succeeded', updated_at=?, artifact_id=?, artifact=? WHERE id=?",
                                (time.time(), job_id, json.dumps(metadata), job_id))
        except Exception:
            self.log_exception("generation failed for " + job_id)
            for path in (temporary, destination):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    self.log_exception("artifact cleanup failed for " + job_id)
            with self.condition, self.db:
                self.db.execute("UPDATE jobs SET status='failed', updated_at=?, error='generation failed', artifact_id=NULL, artifact=NULL WHERE id=?", (time.time(), job_id))

    def _run(self):
        while True:
            try:
                with self.condition:
                    if self.stopping:
                        return
                    job = self._claim()
                    if job is None:
                        self.prune()
                        self.condition.wait(timeout=60)
                        continue
                self._generate(job)
            except Exception:
                self.log_exception("worker loop failed")
                with self.condition:
                    if not self.stopping:
                        self.condition.wait(timeout=5)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                return  # Daemon exits with process; startup recovery marks interrupted jobs.
        self.db.close()


def make_server(worker: Worker, host: str = "0.0.0.0", port: int = 8001) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AudioWorker"
        sys_version = ""

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, _format, *args):
            pass  # Request paths/headers may contain credentials. Never log them.

        def reply(self, status: int, value: dict):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(body)

        def dispatch(self):
            try:
                if self.command == "GET" and self.path == "/healthz":
                    self.reply(200, {"status": "ok"})
                    return
                supplied = self.headers.get_all("Authorization", [])
                expected = ("Bearer " + worker.config.token).encode("ascii")
                if len(supplied) != 1 or not hmac.compare_digest(supplied[0].encode("utf-8"), expected):
                    raise APIError(401, "unauthorized")
                if self.command == "GET" and self.path == "/models":
                    self.reply(200, {"models": [{"id": model, "kind": kind, "license": LICENSE} for kind, model in MODELS.items()]})
                elif self.command == "POST" and self.path == "/jobs":
                    if self.headers.get_content_type() != "application/json":
                        raise APIError(415, "Content-Type must be application/json")
                    lengths = self.headers.get_all("Content-Length", [])
                    if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                        raise APIError(400, "a single Content-Length is required")
                    if len(lengths[0]) > 5:
                        raise APIError(413, "JSON body must be 1..16384 bytes")
                    length = int(lengths[0])
                    if not 1 <= length <= MAX_BODY:
                        raise APIError(413, "JSON body must be 1..16384 bytes")
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise APIError(400, "incomplete JSON body")
                    try:
                        payload = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()), object_pairs_hook=unique_object)
                    except (ValueError, UnicodeError, RecursionError):
                        raise APIError(400, "invalid JSON") from None
                    self.reply(202, worker.submit(payload))
                elif self.command == "GET" and self.path.startswith("/jobs/"):
                    self.reply(200, worker.get_job(self.path[len("/jobs/"):]))
                elif self.command == "GET" and self.path.startswith("/artifacts/"):
                    self.reply(200, worker.artifact(self.path[len("/artifacts/"):]))
                else:
                    raise APIError(404, "route not found")
            except APIError as exc:
                self.reply(exc.status, {"error": str(exc)})
            except (TimeoutError, ConnectionError, OSError):
                self.close_connection = True
            except Exception:
                worker.log_exception("request failed")
                self.reply(500, {"error": "internal worker error"})

        do_GET = dispatch
        do_POST = dispatch

    return ThreadingHTTPServer((host, port), Handler)


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        config = Config.from_env()
        port = int(os.environ.get("WORKER_PORT", "8001"))
        if not 1 <= port <= 65535:
            raise ValueError("WORKER_PORT must be between 1 and 65535")
    except (ValueError, OSError):
        LOGGER.error("invalid worker startup configuration; check NC opt-in, token file, limits and port")
        raise SystemExit(1) from None
    worker = Worker(config)
    server = make_server(worker, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()


if __name__ == "__main__":
    main()
