"""Run: python -m unittest discover -s tests -p test_gateway.py -v"""

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
import wave

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER = Path(__file__).resolve().parents[1] / "gateway" / "server.py"
spec = importlib.util.spec_from_file_location("gateway_server", SERVER)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
TOKEN = "test-gateway-token-" + "x" * 32
WORKER_TOKEN = "test-worker-token-" + "y" * 32


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "mcp_token").write_text(TOKEN, encoding="utf-8")
        (self.root / "worker_token").write_text(WORKER_TOKEN, encoding="utf-8")
        self.env = patch.dict(os.environ, {
            "MCP_TOKEN_FILE": str(self.root / "mcp_token"),
            "WORKER_TOKEN_FILE": str(self.root / "worker_token"),
            "OUTPUT_DIR": str(self.root), "WORKER_URL": "http://inference:8001",
            "PUBLIC_BASE_URL": "http://127.0.0.1:8000", "MCP_ALLOWED_HOSTS": "",
            "MCP_ALLOWED_ORIGINS": "",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.job_id, self.artifact_id = str(uuid4()), str(uuid4())
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x01\x00" * 16000)
        self.wav = audio.getvalue()
        (self.root / f"{self.artifact_id}.wav").write_bytes(self.wav)
        self.artifact = {
            "id": self.artifact_id, "filename": f"{self.artifact_id}.wav", "media_type": "audio/wav",
            "size_bytes": len(self.wav), "duration": 1.0, "sample_rate": 16000,
            "model": "fixed-test-model", "seed": 0, "license": "CC-BY-NC-4.0",
        }
        self.failure = None
        self.requests = []
        self.app = server.create_app(worker_transport=httpx.MockTransport(self.worker))
        self.started, self.stopped = asyncio.Event(), asyncio.Event()

        async def run_lifespan():
            async with self.app.router.lifespan_context(self.app):
                self.started.set()
                await self.stopped.wait()

        self.lifespan_task = asyncio.create_task(run_lifespan())
        await self.started.wait()

        async def stop_lifespan():
            self.stopped.set()
            await self.lifespan_task

        self.addAsyncCleanup(stop_lifespan)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                        base_url="http://127.0.0.1:8000")
        self.addAsyncCleanup(self.client.aclose)

    async def worker(self, request):
        self.assertEqual(request.headers["authorization"], f"Bearer {WORKER_TOKEN}")
        self.requests.append(request)
        if self.failure == "timeout":
            raise httpx.ReadTimeout("sensitive worker detail", request=request)
        if self.failure == "connect":
            raise httpx.ConnectError("sensitive worker detail", request=request)
        if self.failure == "invalid-json":
            return httpx.Response(200, text="not JSON")
        if isinstance(self.failure, int):
            return httpx.Response(self.failure, text="sensitive worker detail")
        if request.url.path == "/models":
            return httpx.Response(200, json={"models": [
                {"id": "fixed-test-model", "kind": kind, "license": "CC-BY-NC-4.0"}
                for kind in ("music", "sound")
            ]})
        if request.url.path == "/jobs" and request.method == "POST":
            return httpx.Response(202, json={"id": self.job_id, "status": "queued"})
        if request.url.path == f"/jobs/{self.job_id}":
            return httpx.Response(200, json={"id": self.job_id, "status": "succeeded", "kind": "music",
                "prompt": "test", "duration": 1, "seed": 0, "created_at": "now", "updated_at": "now",
                "error": None, "artifact_id": self.artifact_id})
        if request.url.path == f"/artifacts/{self.artifact_id}":
            return httpx.Response(200, json=self.artifact)
        return httpx.Response(404)

    @asynccontextmanager
    async def session(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            headers={"Authorization": f"Bearer {TOKEN}"}, follow_redirects=True,
        ) as client:
            async with streamable_http_client("http://127.0.0.1:8000/mcp/", http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def test_authentication_covers_routes_redirects_and_duplicates(self):
        for path in ("/mcp", "/mcp/", f"/artifacts/{self.artifact_id}", "/missing"):
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                self.assertEqual((await self.client.get(path, headers=headers)).status_code, 401)
        duplicate = [("Authorization", f"Bearer {TOKEN}"), ("Authorization", f"Bearer {TOKEN}")]
        self.assertEqual((await self.client.get("/mcp", headers=duplicate)).status_code, 401)
        # /mcp is served directly: a 307 to http://host/mcp/ (scheme lost behind the tunnel)
        # makes clients drop Authorization, which broke the Cloudflare MCP portal.
        auth = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json, text/event-stream"}
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        for path in ("/mcp", "/mcp/"):
            response = await self.client.post(path, headers=auth, json=body)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("generate_music", response.text)
            self.assertNotIn("location", response.headers)
        self.assertEqual((await self.client.get("/healthz")).json(), {"gateway": "ok", "upstream": "ok"})

    async def test_sdk_lifecycle_tools_and_artifact(self):
        async with self.session() as session:
            tools = await session.list_tools()
            self.assertEqual({tool.name for tool in tools.tools}, {
                "generate_music", "generate_sound", "get_job", "get_artifact", "list_models"})
            for kind in ("music", "sound"):
                result = await session.call_tool(f"generate_{kind}", {"prompt": "test", "duration": 1, "seed": 0})
                self.assertFalse(result.isError, result)
                self.assertEqual(json.loads(self.requests[-1].content)["kind"], kind)
            result = await session.call_tool("get_job", {"job_id": self.job_id})
            self.assertFalse(result.isError, result)
            result = await session.call_tool("get_artifact", {"artifact_id": self.artifact_id})
            self.assertFalse(result.isError, result)
            self.assertIn(f"http://127.0.0.1:8000/artifacts/{self.artifact_id}", str(result))
            self.assertFalse((await session.call_tool("list_models", {})).isError)
            for duration, seed in ((0.5, 0), (10, 2**32 - 1)):
                result = await session.call_tool("generate_music", {"prompt": "x", "duration": duration, "seed": seed})
                self.assertFalse(result.isError, result)
        response = await self.client.get(f"/artifacts/{self.artifact_id}", headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, self.wav)

    async def test_invalid_generation_arguments_never_reach_worker(self):
        cases = [
            {"prompt": ""}, {"prompt": "x" * 2001}, {"prompt": 3},
            {"prompt": "x", "duration": 0.49}, {"prompt": "x", "duration": 10.01},
            {"prompt": "x", "duration": True}, {"prompt": "x", "duration": "1"},
            {"prompt": "x", "seed": True}, {"prompt": "x", "seed": -1},
            {"prompt": "x", "seed": 2**32}, {"prompt": "x", "seed": 1.5},
            {"prompt": "x", "model": "arbitrary"}, {"prompt": "x", "url": "http://evil"},
            {"prompt": "x", "path": "/secrets"},
        ]
        async with self.session() as session:
            for args in cases:
                with self.subTest(args=args):
                    result = await session.call_tool("generate_music", args)
                    self.assertTrue(result.isError, result)
        self.assertEqual(self.requests, [])
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaises(Exception):
                await self.app.state.mcp.call_tool("generate_music", {"prompt": "x", "duration": value})
        self.assertEqual(self.requests, [])

    async def test_canonical_ids_missing_and_traversal(self):
        async with self.session() as session:
            for value in ("../secret", self.job_id.upper(), self.job_id.replace("-", ""), "", "http://example.com"):
                for tool, key in (("get_job", "job_id"), ("get_artifact", "artifact_id")):
                    self.assertTrue((await session.call_tool(tool, {key: value})).isError)
            self.assertTrue((await session.call_tool("get_job", {"job_id": str(uuid4())})).isError)
        self.assertEqual(len(self.requests), 1)
        response = await self.client.get("/artifacts/not-a-uuid", headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(response.status_code, 400)
        for filename in ("../secret.wav", f"sub/{self.artifact_id}.wav", f"{uuid4()}.wav"):
            self.artifact["filename"] = filename
            response = await self.client.get(f"/artifacts/{self.artifact_id}", headers={"Authorization": f"Bearer {TOKEN}"})
            self.assertEqual(response.status_code, 502)

    async def test_missing_and_symlink_files_rejected(self):
        path = self.root / f"{self.artifact_id}.wav"
        path.unlink()
        headers = {"Authorization": f"Bearer {TOKEN}"}
        self.assertEqual((await self.client.get(f"/artifacts/{self.artifact_id}", headers=headers)).status_code, 404)
        outside = self.root / "outside.wav"
        outside.write_bytes(self.wav)
        try:
            path.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc.strerror}")
        self.assertEqual((await self.client.get(f"/artifacts/{self.artifact_id}", headers=headers)).status_code, 404)

    async def test_worker_failures_are_clean_and_health_is_degraded(self):
        async with self.session() as session:
            for failure in (429, 503, 500, "timeout", "connect", "invalid-json"):
                self.failure = failure
                result = await session.call_tool("generate_music", {"prompt": "test"})
                self.assertTrue(result.isError)
                self.assertNotIn("sensitive worker detail", str(result))
                self.assertNotIn(WORKER_TOKEN, str(result))
                self.assertIn("worker" if failure != 429 else "queue", str(result))
                response = await self.client.get("/healthz")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"gateway": "ok", "upstream": "unavailable"})

    async def test_mcp_body_limit_headers_chunks_and_auth_order(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = await self.client.post("/mcp/", headers={**headers, "Content-Length": str(65537)}, content=b"{}")
        self.assertEqual(response.status_code, 413)

        async def chunks():
            yield b"x" * 32768
            yield b"x" * 32769

        response = await self.client.post("/mcp/", headers=headers, content=chunks())
        self.assertEqual(response.status_code, 413)
        consumed = False

        async def forbidden_body():
            nonlocal consumed
            consumed = True
            yield b"x" * 65537

        response = await self.client.post("/mcp/", content=forbidden_body())
        self.assertEqual(response.status_code, 401)
        self.assertFalse(consumed)
        self.assertEqual(self.requests, [])
        async with self.session() as session:
            self.assertFalse((await session.call_tool("list_models", {})).isError)

    async def test_dns_rebinding_protection(self):
        response = await self.client.post("/mcp/", headers={"Authorization": f"Bearer {TOKEN}",
            "Host": "attacker.example", "Content-Type": "application/json"}, json={})
        self.assertEqual(response.status_code, 421)
        response = await self.client.post("/mcp/", headers={"Authorization": f"Bearer {TOKEN}",
            "Origin": "https://attacker.example", "Content-Type": "application/json"}, json={})
        self.assertEqual(response.status_code, 403)

    async def test_smoke_script_against_fake_worker(self):
        from types import SimpleNamespace
        smoke_path = SERVER.parents[1] / "scripts" / "smoke.py"
        smoke_spec = importlib.util.spec_from_file_location("gateway_smoke", smoke_path)
        smoke = importlib.util.module_from_spec(smoke_spec)
        smoke_spec.loader.exec_module(smoke)
        original_client = httpx.AsyncClient

        def client(**kwargs):
            return original_client(transport=httpx.ASGITransport(app=self.app), **kwargs)

        args = SimpleNamespace(token_file=self.root / "mcp_token", scheme="http", host="127.0.0.1",
                               port=8000, timeout=5, poll_interval=0.001)
        with patch.object(smoke.httpx, "AsyncClient", side_effect=client):
            await smoke.check(args, self.root)
        submitted = [json.loads(request.content) for request in self.requests if request.method == "POST"]
        self.assertEqual([body["kind"] for body in submitted], ["music", "sound"])
        self.assertEqual([body["duration"] for body in submitted], [1, 1])
        self.assertEqual(len(list(self.root.glob("music-*.wav"))), 1)
        self.assertEqual(len(list(self.root.glob("sound-*.wav"))), 1)
        with self.assertRaises(smoke.SmokeError):
            smoke.validate_wav(b"not WAV")

    async def test_short_token_fails_startup(self):
        (self.root / "mcp_token").write_text("short", encoding="utf-8")
        app = server.create_app(worker_transport=httpx.MockTransport(self.worker))
        with self.assertRaisesRegex(RuntimeError, "at least 32"):
            async with app.router.lifespan_context(app):
                pass


if __name__ == "__main__":
    unittest.main()
