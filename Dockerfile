# Kidney-RAG — Hugging Face Spaces (Docker SDK) image.
#
# HF Spaces expects the app to listen on $PORT (usually 7860). Everything else
# is standard: install deps, copy the repo, start uvicorn.
#
# The MedEmbed model (~1.3 GB) is downloaded on first request and cached to
# /data (HF Spaces persistent volume when enabled) or /tmp/hf_cache otherwise.

FROM python:3.11-slim

# System deps: build tools for a couple of C-extension wheels, plus libgomp
# which faiss / sentence-transformers pull in transitively.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HF Spaces runs as a non-root user (uid 1000). Give it a writable HOME
# and point HuggingFace cache into a directory it owns.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860

RUN useradd -m -u 1000 user

# Install Python deps first so Docker caches the layer across code changes.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo.
COPY --chown=user:user . .

USER user
EXPOSE 7860

CMD ["sh", "-c", "uvicorn web.backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
