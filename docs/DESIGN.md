# Aporia — Design

**One-liner:** search 2,000 years of philosophy by argument, not keyword. Type a
claim — *"free will is an illusion"* — and get the actual passages where
philosophers argued for it and against it, with citations.

## The core technical problem

Ordinary semantic search retrieves passages that are *topically* similar to the
query. That's necessary but not sufficient: it can't tell a passage defending
free will from one attacking it, because both are about free will and embed
near each other in vector space.

So the system has two jobs:

1. **Retrieval** — find topically relevant passages fast (vector search).
2. **Stance + argument understanding** — determine, for each retrieved
   passage, what it argues *relative to the query* (for / against / reframing),
   and summarize the move it makes.

The second job is the differentiator: stance-aware retrieval over dense
primary texts is a genuinely hard problem that off-the-shelf embeddings don't
solve.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
OFFLINE (batch)  │ Ingestion → Chunking → Embedding → Index      │
                 └──────────────────────────────────────────────┘
                                 │ artifacts (index + SQLite)
                                 ▼
ONLINE (serving)  query → embed → vector search (top-K) →
                  stance classification + move summary (LLM, cached) →
                  group into FOR / AGAINST / NUANCE → JSON → web UI
```

Everything heavy happens offline; serving is a vector lookup plus one batched,
aggressively cached LLM call. That keeps a live demo cheap and always-on.

### Ingestion (`ingest/`)
- Public-domain primary texts from Project Gutenberg, boilerplate stripped.
- Structural metadata (author, work, section heading, char offsets) is
  first-class — it's what makes results credible and enables "read in context".

### Chunking (`ingest/chunk.py`)
- ~100–300 word argument units, merged from paragraphs, respecting sentence
  boundaries; section headings tracked heuristically as citation paths.

### Embedding + index (`index/`)
- `BAAI/bge-small-en-v1.5` embeddings (open, CPU-friendly; swap via config).
- hnswlib behind a **swappable `VectorIndex` interface**. The Phase 2 project
  is replacing it with a from-scratch HNSW implementation and benchmarking
  recall/QPS/latency against both the library and the brute-force NumPy
  baseline (already implemented as `NumpyIndex`).

### Stance layer (`api/stance.py`)
- Query-time: one batched Claude call classifies every retrieved passage as
  `for` / `against` / `nuance` with a one-line "move" summary and confidence.
- Results cached in SQLite per (query, passage) — repeat searches skip the LLM
  entirely. Backend failures degrade gracefully (passages shown unclassified,
  never cached).
- Guardrail: every passage shown maps to a real chunk with a real source
  location. The LLM only labels; it never generates quoted text. When unsure
  it labels `nuance` rather than forcing a side.

### API + UI (`api/`, `web/`)
- FastAPI: `GET /search?q=...` → grouped JSON; `GET /passage/{id}` → chunk
  plus neighbors for in-context reading.
- Single-page UI, two-column FOR/AGAINST layout, shareable `?q=` links.

## Phased roadmap

| Phase | Deliverable |
|---|---|
| 0 (done) | Free-will debate, 4 works, working FOR/AGAINST search + UI + evals |
| 1 | 15–25 works across several classic debates; deployed always-on demo |
| 2 | Own HNSW implementation; recall/QPS/latency benchmark table in README |
| 3 | Offline claim graph (supports/attacks relations across works); labeled eval set for retrieval hit-rate and stance accuracy |

## Decisions

- **Corpus = Project Gutenberg public domain only** (licensing-safe to
  redistribute). SEP/PhilPapers are link-out candidates, never bulk-ingested.
- **Library index first, own HNSW later** — ship the product, then earn the
  infra depth. Interface kept stable from day one.
- **Stance at query time, claim graph later** — the graph (Phase 3) moves
  argument extraction offline and makes serving pure lookup.
