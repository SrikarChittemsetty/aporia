# Deploying Aporia

The app is a single FastAPI process that serves the API and the UI. All heavy
compute (embedding the corpus, building the index) happens offline; the server
just loads the artifacts, so a small always-on instance is enough.

## What the server needs

1. The build artifacts: `data/aporia.db` + `data/index/` (run the pipeline
   locally or in CI: `ingest.download` → `ingest.chunk` → `index.build_index`).
2. `ANTHROPIC_API_KEY` set in the environment, so the stance layer uses the
   Anthropic SDK (the `claude` CLI fallback is for local dev only).
3. ~1.5 GB RAM (embedding model + index in memory).

## Fly.io (recommended)

```bash
brew install flyctl && fly auth login
fly launch --name aporia --no-deploy   # generates fly.toml; pick a region
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

Dockerfile sketch:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the corpus + index at image build time (or mount a volume instead)
RUN python -m ingest.download && python -m ingest.chunk && python -m index.build_index
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Pre-download the embedding model in the image too if you want fast cold
starts: `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"`.

## Render / Railway

Same shape: Docker deploy, set `ANTHROPIC_API_KEY`, expose port 8080. Use a
persistent disk or bake artifacts into the image.

## Cost control

- Stance results are cached in SQLite per (query, passage) — each unique
  claim costs one batched Claude call, ever.
- Retrieval is pure CPU. The only recurring cost is the LLM call on novel
  queries; consider capping `k` and rate-limiting `/search` if the demo gets
  traffic.
