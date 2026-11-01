"""Static-bearer MCP gateway; generation stays in the isolated worker."""

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
import stat
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, StrictInt, StrictStr, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

Prompt = Annotated[str, Field(strict=True, min_length=1, max_length=2000)]
Duration = Annotated[float, Field(strict=True, ge=0.5, le=10, allow_inf_nan=False)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class WorkerError(Exception):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


class GatewayMCP(FastMCP):
    async def list_tools(self):
        tools = await super().list_tools()
        for tool in tools:
            tool.inputSchema["additionalProperties"] = False
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        for tool in await self.list_tools():
            if tool.name == name and arguments.keys() - tool.inputSchema["properties"].keys():
                raise ToolError("Unexpected tool arguments; URLs, paths, and model selection are not supported.")
        return await super().call_tool(name, arguments)


class Artifact(BaseModel):
    id: StrictStr
    filename: StrictStr
    media_type: Literal["audio/wav"]
    size_bytes: Annotated[StrictInt, Field(gt=0)]
    duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    sample_rate: Annotated[StrictInt, Field(gt=0)]
    model: StrictStr
    seed: Seed
    license: Literal["CC-BY-NC-4.0"]


def canonical_uuid(value: str) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ToolError("Expected a canonical lowercase UUID.") from None
    return value


def read_token(path: str, name: str) -> str:
    try:
        token = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise RuntimeError(f"Cannot read {name} token file.") from None
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise RuntimeError(f"{name} token must contain at least 32 ASCII characters without whitespace.")
    return token


def configured_url(name: str, default: str) -> str:
    value = os.environ.get(name, default).rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path):
        raise RuntimeError(f"{name} must be an HTTP(S) origin without credentials or a path.")
    return value


class StaticBearerAuth:
    """Authenticate before routing, including redirects and unknown routes; no OAuth."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/mcp":
            # Serve /mcp directly. Starlette would 307 to http://host/mcp/ (the tunnel hides
            # the original https scheme), and clients drop Authorization on that redirect.
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
        if scope["type"] == "http" and scope["path"] != "/healthz":
            headers = [value for key, value in scope["headers"] if key.lower() == b"authorization"]
            expected = b"Bearer " + scope["app"].state.gateway_token.encode("ascii")
            if len(headers) != 1 or not hmac.compare_digest(headers[0], expected):
                response = JSONResponse({"error": "Unauthorized"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
            if scope["path"] == "/mcp" or scope["path"].startswith("/mcp/"):
                limit = 64 * 1024
                lengths = [value for key, value in scope["headers"] if key.lower() == b"content-length"]
                try:
                    if any(int(value) > limit or int(value) < 0 for value in lengths):
                        raise ValueError
                except ValueError:
                    await JSONResponse({"error": "MCP request body exceeds 64 KiB or has invalid length."}, status_code=413)(scope, receive, send)
                    return
                chunks, size = [], 0
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > limit:
                        await JSONResponse({"error": "MCP request body exceeds 64 KiB."}, status_code=413)(scope, receive, send)
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
                body = b"".join(chunks)
                original_receive, replayed = receive, False

                async def bounded_receive():
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await original_receive()

                receive = bounded_receive
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)


def create_app(*, worker_transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    worker_url = configured_url("WORKER_URL", "http://inference:8001")
    public_url = configured_url("PUBLIC_BASE_URL", "http://127.0.0.1:8000")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data")).resolve()
    extra_hosts = [v.strip() for v in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if v.strip()]
    if any("/" in v or "@" in v or "*" in v.removesuffix(":*") for v in extra_hosts):
        raise RuntimeError("MCP_ALLOWED_HOSTS requires explicit hostnames, optionally with a port or :*.")
    extra_origins = [v.strip() for v in os.environ.get("MCP_ALLOWED_ORIGINS", "").split(",") if v.strip()]
    mcp = GatewayMCP(
        "AudioCraft", stateless_http=True, json_response=True, streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", *extra_hosts],
            allowed_origins=["http://localhost:*", "http://127.0.0.1:*", *extra_origins],
        ),
    )
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.gateway_token = read_token(os.environ.get("MCP_TOKEN_FILE", "/run/secrets/mcp_token"), "MCP")
        worker_token = read_token(os.environ.get("WORKER_TOKEN_FILE", "/run/secrets/worker_token"), "Worker")
        async with httpx.AsyncClient(
            base_url=worker_url, headers={"Authorization": f"Bearer {worker_token}"},
            timeout=httpx.Timeout(15, connect=3), transport=worker_transport,
            follow_redirects=False, trust_env=False,
        ) as client:
            app.state.worker = client
            async with mcp.session_manager.run():
                yield

    async def worker(method: str, path: str, *, body: dict | None = None) -> dict:
        try:
            response = await app.state.worker.request(method, path, json=body)
        except httpx.TimeoutException:
            raise WorkerError("Inference worker timed out; retry later.") from None
        except httpx.RequestError:
            raise WorkerError("Inference worker is unavailable; retry later.") from None
        if response.status_code == 429:
            raise WorkerError("Inference queue is full; retry later.", 429)
        if response.status_code == 404:
            raise WorkerError("Job or artifact not found.", 404)
        if response.status_code == 503:
            raise WorkerError("Inference worker is unavailable; retry later.")
        if response.status_code not in {200, 202}:
            raise WorkerError("Inference worker rejected the request.", 502)
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError
            return data
        except ValueError:
            raise WorkerError("Inference worker returned an invalid response.", 502) from None

    async def tool_worker(method: str, path: str, *, body: dict | None = None) -> dict:
        try:
            return await worker(method, path, body=body)
        except WorkerError as exc:
            raise ToolError(str(exc)) from None

    async def submit(kind: str, prompt: str, duration: float, seed: int) -> dict:
        data = await tool_worker("POST", "/jobs", body={"kind": kind, "prompt": prompt, "duration": duration, "seed": seed})
        canonical_uuid(data.get("id"))
        if not isinstance(data.get("status"), str):
            raise ToolError("Inference worker returned an invalid job.")
        return {"id": data["id"], "status": data["status"]}

    @mcp.tool()
    async def generate_music(prompt: Prompt, duration: Duration = 5, seed: Seed = 0) -> dict[str, Any]:
        """Queue non-commercial music generation (0.5–10 seconds). Returns a job ID."""
        return await submit("music", prompt, duration, seed)

    @mcp.tool()
    async def generate_sound(prompt: Prompt, duration: Duration = 5, seed: Seed = 0) -> dict[str, Any]:
        """Queue non-commercial sound generation (0.5–10 seconds). Returns a job ID."""
        return await submit("sound", prompt, duration, seed)

    @mcp.tool()
    async def get_job(job_id: str) -> dict[str, Any]:
        """Poll a generation job using its canonical UUID."""
        job_id = canonical_uuid(job_id)
        data = await tool_worker("GET", f"/jobs/{job_id}")
        if data.get("id") != job_id or not isinstance(data.get("status"), str):
            raise ToolError("Inference worker returned an invalid job.")
        if data.get("artifact_id") is not None:
            canonical_uuid(data["artifact_id"])
        if data.get("error") is not None:
            data["error"] = "Generation failed; inspect worker logs."
        return data

    async def artifact_metadata(artifact_id: str) -> dict:
        data = await worker("GET", f"/artifacts/{artifact_id}")
        try:
            metadata = Artifact.model_validate(data)
            if metadata.id != artifact_id or metadata.filename != f"{artifact_id}.wav":
                raise ValueError
        except (ValidationError, ValueError):
            raise WorkerError("Inference worker returned invalid artifact metadata.", 502) from None
        return metadata.model_dump()

    @mcp.tool()
    async def get_artifact(artifact_id: str) -> dict[str, Any]:
        """Return WAV metadata and an authenticated download URL; never inline audio."""
        artifact_id = canonical_uuid(artifact_id)
        try:
            metadata = await artifact_metadata(artifact_id)
        except WorkerError as exc:
            raise ToolError(str(exc)) from None
        return {**metadata, "download_url": f"{public_url}/artifacts/{artifact_id}"}

    @mcp.tool()
    async def list_models() -> dict[str, Any]:
        """List the worker's fixed model inventory and licenses; no model selection."""
        data = await tool_worker("GET", "/models")
        models = data.get("models")
        if not isinstance(models, list) or any(
            not isinstance(model, dict) or model.get("kind") not in {"music", "sound"}
            or any(not isinstance(model.get(key), str) for key in ("id", "license")) for model in models
        ):
            raise ToolError("Inference worker returned an invalid model inventory.")
        return {"models": [{key: model[key] for key in ("id", "kind", "license")} for model in models]}

    async def health(_: Request) -> JSONResponse:
        try:
            await worker("GET", "/models")
        except WorkerError:
            return JSONResponse({"gateway": "ok", "upstream": "unavailable"}, status_code=503)
        return JSONResponse({"gateway": "ok", "upstream": "ok"})

    async def download(request: Request):
        try:
            artifact_id = canonical_uuid(request.path_params["artifact_id"])
        except ToolError:
            return JSONResponse({"error": "Invalid artifact UUID."}, status_code=400)
        try:
            metadata = await artifact_metadata(artifact_id)
        except WorkerError as exc:
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        path = output_dir / metadata["filename"]
        try:
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                    or path.resolve(strict=True).parent != output_dir):
                raise OSError
            if info.st_size != metadata["size_bytes"]:
                raise OSError
        except OSError:
            return JSONResponse({"error": "Artifact file is unavailable."}, status_code=404)
        return FileResponse(path, media_type="audio/wav", filename=metadata["filename"], stat_result=info)

    app = Starlette(
        routes=[Route("/healthz", health), Route("/artifacts/{artifact_id}", download), Mount("/mcp", app=mcp_app)],
        middleware=[Middleware(StaticBearerAuth)], lifespan=lifespan,
    )
    app.state.mcp = mcp
    return app


app = create_app()
