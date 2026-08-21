# Kidney-RAG — production Docker image.
#
# Targets Azure Container Apps (also runs on any Docker-native host: Cloud Run,
# Fly, Oracle A1, local docker run). Listens on $PORT (default 8000).
#
# The MedEmbed model (~1.3 GB) downloads on first request into
# /home/user/.cache/huggingface and is cached across requests but NOT across
# container restarts — Azure Container Apps has ephemeral disk. First cold
# start is ~30-40s while the model downloads; every subsequent request is fast.
# Keep min replicas >= 1 to avoid re-downloading on every scale-out.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN useradd -m -u 1000 user

# Install Python deps first so Docker caches this layer across code changes.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
EXPOSE 8000

CMD ["sh", "-c", "uvicorn web.backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
