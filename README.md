# Aporia

**Search 2,000 years of philosophy by argument, not keyword.**

Type a claim — *"free will is an illusion"* — and get the actual
primary-source passages where philosophers argued **for** and **against** it,
with citations and a one-line summary of the move each passage makes.

![Aporia searching "free will is an illusion": Spinoza argues FOR, William James and Kant argue AGAINST](docs/screenshot.png)

*Above: Spinoza lands on FOR ("suspension of judgment is a perception, and not
free will"), James and Kant on AGAINST — retrieved from the original texts,
classified by stance, cited to the section.*

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

The three example queries on the landing page ship with pre-cached stance
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
- **`api/`** — FastAPI. `/search` does retrieval + stance grouping;
  `/passage/{id}` returns a chunk with its neighbors. The stance layer
  ([api/stance.py](api/stance.py)) batches all passages into one Claude call,
  caches per (query, passage) in SQLite, and degrades gracefully (unclassified
  results, never cached) if no LLM backend is reachable.
- **`evals/`** — retrieval sanity suite: 10 hand-written claims with expected
  authors. Current score: **9/10 queries surface an expected author in the
  top 12** over a 1,486-chunk corpus.

## Corpus (Phase 0: the free-will debate)

All public domain, via Project Gutenberg:

| Author | Work | Position |
|---|---|---|
| David Hume | An Enquiry Concerning Human Understanding | compatibilist — "Of Liberty and Necessity" |
| Baruch Spinoza | Ethics | necessitarian — the feeling of freedom is ignorance of causes |
| William James | The Will to Believe | indeterminist — "The Dilemma of Determinism" |
| Immanuel Kant | Critique of Practical Reason | freedom as a postulate of practical reason |

## Roadmap

- **Phase 1** — 15–25 works across several classic debates (personal identity,
  objective morality, the existence of God); deployed always-on demo
  ([DEPLOY.md](DEPLOY.md) has the Fly.io recipe).
- **Phase 2** — replace hnswlib with a from-scratch HNSW implementation behind
  the same interface; benchmark recall vs. the brute-force baseline and
  QPS/latency vs. the library, table goes here.
- **Phase 3** — offline claim graph: extract structured claims and
  supports/attacks relations across works, so serving becomes graph traversal
  instead of query-time classification; hand-labeled eval set for stance
  accuracy.

## License

MIT — see [LICENSE](LICENSE). Corpus texts are public domain.
