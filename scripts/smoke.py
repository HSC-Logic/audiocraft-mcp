"""Authenticated real-model smoke check. Never accepts a token value on the CLI."""

import argparse
import asyncio
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlsplit
from uuid import UUID
import wave

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class SmokeError(Exception):
    pass


def result_object(result) -> dict:
    if result.isError:
        # SDK errors may include arguments; do not echo them or upstream bodies.
        raise SmokeError("MCP tool failed; inspect gateway/worker logs.")
    data = result.structuredContent
    if isinstance(data, dict):
        return data
    for block in result.content:
        if block.type == "text":
            try:
                data = json.loads(block.text)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    raise SmokeError("MCP tool returned no JSON object.")


def uuid_value(value) -> str:
    if not isinstance(value, str):
        raise SmokeError("Worker returned an invalid ID.")
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise SmokeError("Worker returned an invalid ID.") from None
    return value


def validate_wav(data: bytes) -> tuple[int, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getnchannels() < 1 or wav.getsampwidth() < 1:
                raise SmokeError("Downloaded WAV has invalid PCM parameters.")
            frames, rate = wav.getnframes(), wav.getframerate()
            samples = wav.readframes(frames)
            if frames < 1 or rate < 1 or len(samples) != frames * wav.getnchannels() * wav.getsampwidth():
                raise SmokeError("Downloaded WAV is empty or truncated.")
            return frames, rate
    except (wave.Error, EOFError):
        raise SmokeError("Downloaded artifact is not a readable WAV.") from None


async def check(args, output_dir: Path) -> None:
    try:
        token = args.token_file.read_text(encoding="utf-8").strip()
    except OSError:
        raise SmokeError("Cannot read token file; set --token-file to .secrets/mcp_token or your mounted secret.") from None
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise SmokeError("Token file must contain at least 32 ASCII characters without whitespace.")
    base_url = f"{args.scheme}://{args.host}:{args.port}"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as http:
        if (await http.post(f"{base_url}/mcp/", json={})).status_code != 401:
            raise SmokeError("Unauthenticated MCP request was not rejected with 401.")
        health = await http.get(f"{base_url}/healthz")
        if health.status_code != 200 or health.json().get("upstream") != "ok":
            raise SmokeError("Gateway or worker health failed; start both services and inspect /healthz.")
        async with (
            httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(30, read=300), trust_env=False) as mcp_http,
            streamable_http_client(f"{base_url}/mcp/", http_client=mcp_http) as (read, write, _),
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                if tools != {"generate_music", "generate_sound", "get_job", "get_artifact", "list_models"}:
                    raise SmokeError("Gateway tool inventory does not match the supported contract.")
                models = result_object(await session.call_tool("list_models", {}))
                if {model["kind"] for model in models.get("models", [])} != {"music", "sound"}:
                    raise SmokeError("Worker model inventory lacks music or sound.")
                for kind, prompt in (("music", "A short gentle piano melody"), ("sound", "A dog barking")):
                    async with asyncio.timeout(args.timeout):
                        job = result_object(await session.call_tool(f"generate_{kind}", {
                            "prompt": prompt, "duration": 1, "seed": 0,
                        }))
                        job_id = uuid_value(job.get("id"))
                        deadline = time.monotonic() + args.timeout
                        while True:
                            job = result_object(await session.call_tool("get_job", {"job_id": job_id}))
                            if job.get("status") == "succeeded":
                                break
                            if job.get("status") == "failed":
                                raise SmokeError(f"{kind} generation failed; inspect worker logs.")
                            if time.monotonic() >= deadline:
                                raise SmokeError(f"{kind} generation timed out; increase --timeout for cold model loading.")
                            await asyncio.sleep(args.poll_interval)
                        artifact_id = uuid_value(job.get("artifact_id"))
                        metadata = result_object(await session.call_tool("get_artifact", {"artifact_id": artifact_id}))
                        download_url = metadata.get("download_url", "")
                        parsed, expected = urlsplit(download_url), urlsplit(base_url)
                        # Never forward a credential to an origin supplied by a worker/server response.
                        if ((parsed.scheme, parsed.hostname, parsed.port) != (expected.scheme, expected.hostname, expected.port)
                                or parsed.path != f"/artifacts/{artifact_id}" or parsed.query or parsed.fragment
                                or parsed.username or parsed.password):
                            raise SmokeError("Artifact URL differs from the tested gateway origin; check PUBLIC_BASE_URL.")
                        if (await http.get(download_url)).status_code != 401:
                            raise SmokeError("Unauthenticated artifact download was not rejected with 401.")
                        response = await http.get(download_url, headers=headers)
                        if response.status_code != 200 or response.headers.get("content-type", "").split(";")[0] != "audio/wav":
                            raise SmokeError("Authenticated WAV download failed.")
                        if len(response.content) != metadata.get("size_bytes"):
                            raise SmokeError("Downloaded artifact size differs from metadata.")
                        frames, rate = validate_wav(response.content)
                        (output_dir / f"{kind}-{artifact_id}.wav").write_bytes(response.content)
                        print(f"{kind}: WAV verified, {frames} frames, {rate} Hz; no audio-quality claim.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scheme", choices=("http", "https"), default="http")
    parser.add_argument("--token-file", type=Path, default=Path(os.environ.get("MCP_TOKEN_FILE", ".secrets/mcp_token")))
    parser.add_argument("--timeout", type=float, default=900, help="Per-generation timeout in seconds, including cold loading")
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--output-dir", type=Path, help="Keep downloaded WAVs here; otherwise remove temporary outputs")
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.poll_interval))
            or any(c in args.host for c in "/@?#") or not args.host):
        parser.error("Use a hostname, valid port, and finite positive timeouts.")
    try:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            asyncio.run(check(args, args.output_dir))
        else:
            with tempfile.TemporaryDirectory(prefix="audiocraft-smoke-") as temporary:
                asyncio.run(check(args, Path(temporary)))
    except SmokeError as exc:
        print(f"Smoke failed: {exc}", file=sys.stderr)
        return 1
    except TimeoutError:
        print("Smoke failed: generation timed out; increase --timeout for cold model loading.", file=sys.stderr)
        return 1
    except (Exception, BaseExceptionGroup):
        print("Smoke failed: transport or response error; verify service health, token files, and gateway/worker logs.", file=sys.stderr)
        return 1
    print("Smoke passed: both generation jobs completed; authentication and WAV structure verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
