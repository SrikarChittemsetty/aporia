FROM python:3.12-slim

WORKDIR /app

# hnswlib compiles from source on slim images
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch first — otherwise pip resolves the CUDA build and adds ~3GB
# of NVIDIA libraries to a CPU serving image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model into the image for fast cold starts.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    from config import EMBED_MODEL; SentenceTransformer(EMBED_MODEL)"

# Build corpus + index at image build time unless the build context already
# carries artifacts (data/ is gitignored but not dockerignored, so a local
# build after running the pipeline skips this).
RUN [ -f data/index/meta.json ] || (python -m ingest.download \
    && python -m ingest.chunk && python -m index.build_index)

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
