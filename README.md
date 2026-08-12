# Aporia

**Search 2,000 years of philosophy by argument, not keyword.**

Type a claim — *"free will is an illusion"* — or just a topic — *"free
will"*, *"evil"*, *"existence of god"* — and get the actual primary-source
passages where philosophers argued **for** and **against** it, with citations
and a one-line summary of the move each passage makes. Bare topics are
resolved to the canonical contested claim first (*"existence of god"* →
*"God exists"*), so the FOR/AGAINST split always has a definite thesis.

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
- **`index/`** — `sentence-transformers` embeddings and an hnswlib index
  behind a swappable [`VectorIndex`](index/vector_index.py) interface. A
  brute-force NumPy index ships alongside as the exact-recall baseline.
- **`api/`** — FastAPI. `/search` does claim resolution, retrieval, and
  stance grouping; `/passage/{id}` returns a chunk with its neighbors. Topic
  queries resolve via a built-in claim table, then a cached LLM call
  ([api/claims.py](api/claims.py)). The stance layer
  ([api/stance.py](api/stance.py)) batches all passages into one Claude call,
  caches per (claim, passage) in SQLite, and degrades gracefully (unclassified
  results, never cached) if no LLM backend is reachable.
- **`evals/`** — retrieval sanity suite: 10 hand-written claims with expected
  authors. Current score: **9/10 queries surface an expected author in the
  top 12** over a 4,252-chunk corpus.

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

## Roadmap

- **Phase 1** — deployed always-on demo ([DEPLOY.md](DEPLOY.md) has the
  Fly.io recipe); more debates (personal identity, beauty, knowledge).
- **Phase 2** — replace hnswlib with a from-scratch HNSW implementation behind
  the same interface; benchmark recall vs. the brute-force baseline and
  QPS/latency vs. the library, table goes here.
- **Phase 3** — offline claim graph: extract structured claims and
  supports/attacks relations across works, so serving becomes graph traversal
  instead of query-time classification; hand-labeled eval set for stance
  accuracy.

## License

MIT — see [LICENSE](LICENSE). Corpus texts are public domain.
