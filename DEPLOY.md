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

The repo ships a [Dockerfile](Dockerfile) that installs CPU-only torch
(avoiding ~3GB of CUDA libraries), bakes the embedding model into the image,
and reuses local `data/` artifacts when present (building corpus + index from
scratch otherwise):

```bash
docker build -t aporia .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... aporia
```

## Render / Railway

Same shape: Docker deploy, set `ANTHROPIC_API_KEY`, expose port 8080. Use a
persistent disk or bake artifacts into the image.

## Cost control

- Stance results are cached in SQLite per (query, passage) — each unique
  claim costs one batched Claude call, ever.
- Retrieval is pure CPU. The only recurring cost is the LLM call on novel
  queries; consider capping `k` and rate-limiting `/search` if the demo gets
  traffic.
