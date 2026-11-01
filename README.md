# AudioCraft MCP

Independent, unofficial HSC-Logic project. Not affiliated with or endorsed by Meta.

AudioCraft MusicGen and AudioGen inference plus an authenticated agent-facing MCP server. **Internal non-commercial testing only; not production-ready.** No commercial backend is included.

## Licensing

Project-authored glue: Apache-2.0. AudioCraft source: MIT. Model weights: CC-BY-NC-4.0. Accepting the startup switch explicitly opts into non-commercial model use; it does not grant commercial rights, guarantee rights to generated content, or replace legal review. See [third-party notices](THIRD_PARTY_NOTICES.md). Models are downloaded at runtime, not bundled in this repository.

## Requirements

- Docker Desktop with Linux containers and WSL2 GPU support on Windows; recent NVIDIA Windows driver. Linux requires Docker Compose and NVIDIA Container Toolkit.
- Target hardware: RTX 4060 Ti 16 GB. CUDA 12.1 / Torch 2.1.0 container GPU detection and a GPU tensor calculation were verified on this host. End-to-end model generation and performance remain unverified until model smoke tests pass. Docker Engine 29.9.2 alone does not prove GPU support.
- Internet access for image builds and initial model downloads; sufficient disk for image layers and model cache.
- Other GPU workloads, including Ollama, compete for VRAM. Do not stop unrelated services automatically. Inspect GPU usage and arrange capacity before generation.

From the repository root, check GPU passthrough before building:

```sh
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## Start

Windows PowerShell:

```powershell
./scripts/setup.ps1 -AcceptNonCommercial
docker compose up -d --build --wait
```

Linux / WSL:

```sh
sh scripts/setup.sh --accept-noncommercial
docker compose up -d --build --wait
```

Omit the acceptance switch to generate configuration without opting into model use. `.env` defaults to `ACCEPT_NONCOMMERCIAL_MODELS=false`; inference refuses unaccepted model use. Setup preserves existing tokens and existing configuration except the explicit acceptance setting. Never commit `.env` or `.secrets/`. Do not copy tokens into issue reports or MCP configuration files.

```sh
docker compose ps
docker compose logs --tail 100 inference gateway
docker compose exec inference python -m pip check
docker compose exec gateway python -m pip check
```

## MCP client

Streamable HTTP endpoint: `http://127.0.0.1:8000/mcp/`. Both MCP and artifact downloads require `Authorization: Bearer <token>`. `/healthz` is an unauthenticated minimal health probe. The default host binding is loopback only; do not expose this testing service publicly.

Example for a client supporting MCP URL configuration and environment substitution:

```json
{
  "mcpServers": {
    "audiocraft": {
      "url": "http://127.0.0.1:8000/mcp/",
      "headers": { "Authorization": "Bearer ${env:AUDIOCRAFT_MCP_TOKEN}" }
    }
  }
}
```

Client configuration schemas differ; use the client's documented Streamable HTTP/custom-header format. This is static bearer authentication, not OAuth. Clients without custom headers are unsupported.

Load the token into the client process environment without printing it:

```powershell
$env:AUDIOCRAFT_MCP_TOKEN = [IO.File]::ReadAllText((Join-Path $PWD '.secrets/mcp_token')).Trim()
# Launch the MCP client from this shell.
```

```sh
export AUDIOCRAFT_MCP_TOKEN="$(cat .secrets/mcp_token)"
# Launch the MCP client from this shell.
```

Generation returns a job ID immediately. Poll job status, then download the artifact using the same bearer token. Do not put tokens in download URLs. Inputs do not accept arbitrary URLs, model IDs, or filesystem paths.

## Runtime boundaries

- `inference`: private port 8001, one GPU, UID 10001. One serialized model worker; MusicGen small and AudioGen medium loaded one at a time. Four queued jobs, one active generation, maximum 10 seconds per output.
- `gateway`: CPU-only UID 10001, published `127.0.0.1:8000`. Official Python SDK `mcp==1.26.0`; no separate FastMCP dependency or Torch stack.
- Persistent `models` volume at `/models`; persistent `outputs` volume at `/data`, read-only in the gateway. Jobs use `/data/jobs.sqlite3`; artifacts use server-generated UUID filenames. Output budget 1 GiB, retention 24 hours.
- Distinct file-mounted worker and external MCP secrets. Inference has no host-published port. Initial downloads require outbound network access.
- AudioCraft 1.3.0 uses Torch 2.1.0 / CUDA 12.1. Worker and gateway dependencies remain isolated. Consult their Dockerfiles and requirement files for exact build pins; dependency resolution and CUDA compatibility need actual build verification.

## Verification and rollback

CPU checks do not establish GPU support:

```sh
python -m compileall -q worker gateway tests scripts
python -m unittest discover -s tests -v
docker compose config --quiet
```

After startup, install the lightweight gateway dependencies into a local virtual environment, then run the smoke test (downloads both models on first use):

```sh
python -m venv .venv
# Windows PowerShell: ./.venv/Scripts/Activate.ps1
# Linux / WSL: . .venv/bin/activate
python -m pip install -r gateway/requirements.lock
python scripts/smoke.py --host 127.0.0.1 --port 8000 --token-file .secrets/mcp_token --timeout 900 --poll-interval 2
```

The smoke test checks MCP initialization/tool listing, both models sequentially, authenticated WAV downloads, and authentication rejection. Separately verify bounded queue, invalid inputs, restart recovery, output expiry, and model-switch VRAM use. Never label these passed without running them.

```sh
docker compose down
# Restore the previously tested source/image version, then rebuild/start.
docker compose up -d --build --wait
```

`down` preserves named volumes. **Never use `down -v` unless intentionally deleting model caches, jobs, and generated artifacts.** Back up output data before upgrades; schema compatibility must be checked before rollback. Rotate tokens by replacing their files securely and recreating both services; old sessions/clients need the replacement token.

See [SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
