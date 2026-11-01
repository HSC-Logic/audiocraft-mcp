FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    OUTPUT_DIR=/data \
    WORKER_PORT=8001

USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev pkg-config build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY worker/requirements.txt worker/constraints.txt /app/
RUN python -m pip install --no-cache-dir -c constraints.txt -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps audiocraft==1.3.0 \
    && out="$(python -m pip check | grep -v '^audiocraft 1.3.0 has requirement av==11.0.0' || true)" \
    && test -z "$out" \
    && python -c "import torch, torchaudio, audiocraft; from audiocraft.models import MusicGen, AudioGen; assert torch.__version__.split('+')[0] == '2.1.0'" \
    && groupadd --gid 10001 worker \
    && useradd --uid 10001 --gid 10001 --create-home worker \
    && mkdir -p /models /data \
    && chown -R 10001:10001 /models /data /app

COPY --chown=10001:10001 worker/server.py /app/server.py
USER 10001:10001
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz', timeout=3)" || exit 1
CMD ["python", "/app/server.py"]
