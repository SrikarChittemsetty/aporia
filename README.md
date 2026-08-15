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
| Retrieval eval | **37 of 41** hand-written claims surface the philosopher who actually holds the position, across all 13 works |
| Stance layer | one Claude call per claim, cached forever |
| Classifier reliability | **88.9%** run-to-run agreement over 144 passages — and **0** for↔against reversals |
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
- **`evals/`** — retrieval sanity suite: 41 hand-written claims with the
  authors who actually argue them, three or four per work. Current score:
  **37/41 (90%) surface an expected author in the top 12**, with passages
  drawn from all 13 works. See the failure mode it exposed, below.
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

## What the bigger eval exposed

Widening the retrieval eval from 10 free-will queries to 41 across every work
barely moved the headline (9/10 → 37/41), which is the boring part. The
interesting part is the four misses, because they are all the same miss.

| query | wanted | got |
|---|---|---|
| "most people mistake shadows for reality" | Plato | Spinoza, Hume, Nietzsche, Descartes, James |
| "morality is an invention of the weak to constrain the strong" | Nietzsche | Kant, Mill |
| "good and evil are historical inventions rather than eternal facts" | Nietzsche | Spinoza, Hume, Kant, Plato, James |
| "we should reject any belief we can find the slightest reason to doubt" | Descartes | Hume, Kant, Mill, James |

Every one of those arguments *is* in the corpus. Re-running the same queries in
the text's own language finds them immediately:

| query | retrieves |
|---|---|
| "most people mistake shadows for reality" | ✗ no Plato |
| "prisoners chained in a cave see only shadows cast on the wall" | ✓ **Plato** |
| "morality is an invention of the weak to constrain the strong" | ✗ no Nietzsche |
| "master morality and slave morality are two distinct types" | ✓ **Nietzsche** |

So the failure mode is specific and diagnosable: **the embedding matches
vocabulary and imagery, not the conclusion an argument reaches.** Plato makes the
point about appearance and reality by telling a story about a den, prisoners and
firelight, and never states the moral in the abstract terms a user would type.
Nietzsche coins his own vocabulary rather than using the paraphrase everyone
remembers him by. Both are exactly the passages a philosophy search engine most
needs to find, and dense retrieval alone reliably misses them.

This is the strongest argument for the Phase 3 work below: extracting the claims
a passage makes, rather than hoping a query happens to share its wording.

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

- `python -m evals.sanity` — retrieval hit-rate on 41 hand-written claims (37/41, 90%)
- `python -m evals.bench` — index recall/latency benchmark (table above)
- `python -m evals.stability` — **run-to-run agreement of the stance
  classifier**: an independent second pass over the same 144 passages, with no
  access to the first pass, compared label by label.

  | | |
  |---|---|
  | agreement | **128/144 = 88.9%** |
  | outright reversals (for ↔ against) | **0** |
  | disagreements | 16, every one of them in or out of *nuance* |

  That second number is the interesting one. The classifier never once flipped
  a passage from defending a claim to attacking it; all of its instability sits
  on the boundary between "takes a side" and "equivocates", which is the
  boundary human readers argue about too. Per-claim agreement ranges from 67%
  (the design argument, where Hume's characters concede and withhold in the
  same breath) to 100% (four of the twelve claims).

  **What this is not:** an accuracy measurement. Both passes come from the same
  model, so this measures self-consistency, not correctness.
- `python -m evals.make_labeling_sheet` → `python -m evals.score_gold` — the
  accuracy eval, and the honest way to get one. The first writes a
  self-contained HTML sheet that shows 60 passages one at a time, stratified
  across claims and shuffled, with **the model's answer nowhere in the file**;
  you label them cold. The second scores the classifier against those labels and
  reports accuracy, a confusion matrix, per-stance precision/recall, and
  **Cohen's kappa** — which is the number to quote, because raw agreement
  flatters any classifier on a skewed label distribution.
- `python -m evals.stance_eval` — agreement against
  [evals/gold_stances.json](evals/gold_stances.json). Read the caveat before
  quoting the result: that file was exported from cached model labels for six
  claims and spot-checked by hand, so it is a **regression baseline** — it
  catches the classifier drifting from behaviour that was once reviewed. It is
  not an independent ground truth, and a true accuracy number needs a human
  labelling passages blind. That is the next eval worth building.

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
