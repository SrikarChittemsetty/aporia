# Aporia

[![tests](https://github.com/SrikarChittemsetty/aporia/actions/workflows/ci.yml/badge.svg)](https://github.com/SrikarChittemsetty/aporia/actions/workflows/ci.yml)

**Search 2,000 years of philosophy by argument, not keyword.**

Type a claim — *"free will is an illusion"* — or just a topic — *"free
will"*, *"evil"*, *"existence of god"* — and get the actual primary-source
passages where philosophers argued **for** and **against** it, with citations
and a one-line summary of the move each passage makes. Bare topics are
resolved to the canonical contested claim first (*"existence of god"* →
*"God exists"*), so the FOR/AGAINST split always has a definite thesis.

### **[▶ Try it — twelve debates, no install](https://srikarchittemsetty.github.io/aporia/)**

Pick a claim and watch Kant and Spinoza land on opposite sides of it. That page
is a frozen snapshot of real pipeline output; the sections below are how it was
built.

| | |
|---|---|
| Corpus | **3,723 passages** from 13 primary works, 10 philosophers |
| Vector index | **written from scratch in NumPy** from the HNSW paper — 0.999 recall@10, 0.46 ms p50 |
| Retrieval eval | **9 of 10** hand-written claims surface the philosopher who actually holds the position |
| Stance layer | one Claude call per claim, cached forever; **83 hand-labelled** gold judgements to check it against |
| Tests | 10 pytest tests, green in CI |

![Aporia searching "free will is an illusion": Spinoza and Nietzsche argue FOR, William James and Kant argue AGAINST](docs/screenshot.png)

*Above: Spinoza and Nietzsche land on FOR, James and Kant on AGAINST —
retrieved from the original texts, classified by stance, cited to the
section.*

## Why this is hard

Ordinary semantic search finds passages *about* a topic. It cannot tell a
defense of free will from an attack on it — both are topically identical and
embed near each other. Aporia layers **stance-aware retrieval** on top of
vector search: every retrieved passage is classified relative to *your claim*
(for / against / nuance) by an LLM, with results cached so each unique claim
is classified exactly once. Full design rationale: [docs/DESIGN.md](docs/DESIGN.md).

```
OFFLINE   Gutenberg texts → clean → chunk (~100–300 words, citation metadata)
          → embed (bge-small) → hnswlib index
ONLINE    claim → embed → top-K vector search → one batched Claude call
          classifies stance + move per passage (cached) → FOR/AGAINST → UI
```

## Try it in two minutes

```bash
git clone https://github.com/SrikarChittemsetty/aporia && cd aporia
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m ingest.download && python -m ingest.chunk && python -m index.build_index
uvicorn api.main:app --port 8080
# open http://localhost:8080 and click an example query
```

The example queries on the landing page ship with pre-cached stance
classifications, so the demo works immediately — no API key needed. Novel
claims use Claude for classification: set `ANTHROPIC_API_KEY` (or be logged
into the `claude` CLI). Without either, novel queries still return passages,
just ungrouped.

Searches are shareable links: `/?q=liberty+is+compatible+with+necessity`.

## What's under the hood

- **`ingest/`** — Project Gutenberg download + cleaning, paragraph-merging
  chunker that preserves author/work/section/char-offset metadata (powers the
  "read in context" expansion).
- **`index/`** — `sentence-transformers` embeddings behind a swappable
  [`VectorIndex`](index/vector_index.py) interface with three interchangeable
  implementations: exact brute force (NumPy), hnswlib, and
  [**a from-scratch HNSW written in pure NumPy**](index/pyhnsw.py) —
  hierarchical proximity graphs, beam search, and the Malkov–Yashunin
  neighbor-selection heuristic implemented from the paper. **The demo serves
  the from-scratch index** (`INDEX_KIND` in config.py switches back to
  hnswlib or exact search). Benchmarks below.
- **`api/`** — FastAPI. `/search` does claim resolution, retrieval, and
  stance grouping; `/passage/{id}` returns a chunk with its neighbors. Topic
  queries resolve via a built-in claim table, then a cached LLM call
  ([api/claims.py](api/claims.py)). The stance layer
  ([api/stance.py](api/stance.py)) batches all passages into one Claude call,
  caches per (claim, passage) in SQLite, and degrades gracefully (unclassified
  results, never cached) if no LLM backend is reachable.
- **`evals/`** — retrieval sanity suite: 10 hand-written claims with expected
  authors. Current score: **9/10 queries surface an expected author in the
  top 12** over the 3,723-chunk corpus.
- **`scripts/export_site.py`** — freezes a curated set of claims into the
  static site at [`docs/`](docs/) that backs the live demo link above. It runs
  the real retrieval path, emits the pipeline's own classification prompt for
  any unclassified (claim, passage) pair, and folds the answers back into the
  same SQLite stance cache the app reads — so the demo is a snapshot of real
  output, and a local run of the app answers those twelve claims with no API
  key. Three stages: `retrieve`, `ingest`, `build`.

## Corpus (13 works, all public domain via Project Gutenberg)

Free will: Hume's *Enquiry*, Spinoza's *Ethics*, James's *The Will to
Believe*, Kant's two *Critiques*-era works. God and evil: Hume's *Dialogues
Concerning Natural Religion*, Descartes's *Discourse on the Method*.
Morality: Nietzsche's *Beyond Good and Evil*, Mill's *Utilitarianism*, Kant's
*Groundwork*. Rights and justice: Locke's *Second Treatise*, Mill's *On
Liberty*, Paine's *Rights of Man*, Plato's *Republic*. The full list with
Gutenberg IDs lives in [config.py](config.py) — adding a work is one dict
entry plus a pipeline re-run (chunking is incremental; existing chunk ids are
stable).

## Index benchmarks

`python -m evals.bench` — 100 held-out corpus vectors as queries, k=10,
recall measured against exact search:

| Index | recall@10 | build time | p50 query | p95 query |
|---|---|---|---|---|
| NumPy brute force (exact) | 1.000 | 0.0s | 0.18 ms | 0.27 ms |
| hnswlib (C++) | 0.996 | 0.3s | 0.20 ms | 0.26 ms |
| PyHNSW (ours, NumPy) | 0.999 | 18.3s | 0.46 ms | 0.58 ms |

*(3,623 vectors, dim 384, MacBook CPU.)*

Honest reading: at this corpus size a single vectorized matrix product is
hard to beat, and C++ beats Python on constant factors. The from-scratch
implementation is the depth exercise — same algorithm, same interface, real
recall — and its per-query work scales ~O(log n) where brute force scales
O(n).

## Tests & evals

`pytest` covers the chunker (heading detection regressions included), claim
resolution, and both pure-Python indexes (recall floor + disk roundtrip).
CI runs on every push.

- `python -m evals.sanity` — retrieval hit-rate on hand-written claims (9/10)
- `python -m evals.bench` — index recall/latency benchmark (table above)
- `python -m evals.stance_eval` — live stance-classifier agreement against
  [83 hand-labeled gold stances](evals/gold_stances.json) (needs an LLM
  backend)

## Roadmap

- **Phase 1** — deployed always-on demo ([DEPLOY.md](DEPLOY.md) has the
  Fly.io recipe); more debates (personal identity, beauty, knowledge).
- **Phase 2** — ✅ from-scratch HNSW ([index/pyhnsw.py](index/pyhnsw.py)),
  benchmarked above and **serving the live demo**; next: tune ef/M
  trade-offs at larger corpus sizes.
- **Phase 3** — offline claim graph: extract structured claims and
  supports/attacks relations across works, so serving becomes graph traversal
  instead of query-time classification; hand-labeled eval set for stance
  accuracy.

## License

MIT — see [LICENSE](LICENSE). Corpus texts are public domain.
