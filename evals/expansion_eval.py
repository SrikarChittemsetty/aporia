"""Does query expansion actually fix the retrieval failure, and what does it cost?

evals/sanity.py found that dense retrieval matches vocabulary rather than
conclusions, and api/expand.py proposes a fix. This measures whether the fix
works, on the same 41 queries, at several blend strengths — including alpha=0,
which is plain retrieval, so the baseline and the treatment come out of the same
run and the same code path.

    python -m evals.expansion_eval --prompts-out data/expansion/prompts
    python -m evals.expansion_eval --ingest data/expansion/passages
    python -m evals.expansion_eval

A fix that helps the four known misses while quietly breaking six other queries
is not a fix, so this reports per-query changes in both directions, not just the
headline rate.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, ROOT, TOP_K
from api import expand
from evals.sanity import QUERIES as SANITY_QUERIES
from index.vector_index import load_index

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

QUERIES = SANITY_QUERIES  # swapped for the held-out set by --holdout


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def stage_prompts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for query, _expected, _topic in QUERIES:
        (out_dir / f"{slug(query)}.txt").write_text(
            expand.PROMPT_TEMPLATE.format(claim=query)
        )
    print(f"wrote {len(QUERIES)} prompts to {out_dir}")
    print("answer them with any Claude backend, save each reply as <same-name>.txt,")
    print("then: python -m evals.expansion_eval --ingest <dir>")


def stage_ingest(in_dir: Path) -> None:
    n = 0
    for query, _expected, _topic in QUERIES:
        path = in_dir / f"{slug(query)}.txt"
        if not path.exists():
            print(f"  (missing) {query}")
            continue
        expand.store(query, path.read_text())
        n += 1
    print(f"stored {n}/{len(QUERIES)} expansions in the cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-out", type=Path, default=None)
    ap.add_argument("--ingest", type=Path, default=None)
    ap.add_argument("--holdout", action="store_true",
                    help="run against evals/holdout.py — queries written after "
                         "the fix and never used to build it")
    args = ap.parse_args()

    global QUERIES
    if args.holdout:
        from evals.holdout import QUERIES as HOLDOUT_QUERIES
        QUERIES = HOLDOUT_QUERIES
        print(f"held-out set: {len(QUERIES)} queries never seen during development\n")

    if args.prompts_out:
        stage_prompts(args.prompts_out)
        return
    if args.ingest:
        stage_ingest(args.ingest)
        return

    import sqlite3

    from sentence_transformers import SentenceTransformer

    from config import DB_PATH

    model = SentenceTransformer(EMBED_MODEL)
    index = load_index(INDEX_DIR)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    def authors_for(vec) -> set[str]:
        ids, _ = index.search(np.asarray(vec, dtype=np.float32), k=TOP_K)
        placeholders = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT author FROM chunks WHERE id IN ({placeholders})", list(ids)
        ).fetchall()
        return {r["author"] for r in rows}

    # Embed once per query and once per expansion; the blend is arithmetic.
    prepared = []
    missing = 0
    for query, expected, topic in QUERIES:
        q_vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        passage = expand.cached(query)
        if passage is None:
            missing += 1
            h_vec = None
        else:
            # The hypothetical is a passage, not a query, so it gets no query
            # prefix — the bge models want that asymmetry.
            h_vec = model.encode(passage, normalize_embeddings=True)
        prepared.append((query, expected, topic, q_vec, h_vec))

    if missing:
        print(f"warning: {missing}/{len(QUERIES)} queries have no cached expansion\n")

    results: dict[float, list[bool]] = {}
    for alpha in ALPHAS:
        hits = []
        for _query, expected, _topic, q_vec, h_vec in prepared:
            vec = q_vec if (h_vec is None or alpha == 0.0) else expand.blend(q_vec, h_vec, alpha)
            hits.append(bool(expected & authors_for(vec)))
        results[alpha] = hits

    n = len(QUERIES)
    print(f"{'alpha':>6} {'hits':>8} {'rate':>8}   (alpha=0 is plain retrieval)")
    print("-" * 44)
    for alpha in ALPHAS:
        h = sum(results[alpha])
        print(f"{alpha:>6.1f} {h:>5}/{n} {h / n:>8.1%}")

    # The honest part: what moved, in both directions.
    base = results[0.0]
    best_alpha = max((a for a in ALPHAS if a > 0), key=lambda a: sum(results[a]))
    best = results[best_alpha]

    fixed = [QUERIES[i][0] for i in range(n) if best[i] and not base[i]]
    broken = [QUERIES[i][0] for i in range(n) if base[i] and not best[i]]

    print(f"\nat the best blend (alpha={best_alpha}): "
          f"{len(fixed)} fixed, {len(broken)} broken")
    for q in fixed:
        print(f"  + {q}")
    for q in broken:
        print(f"  - {q}")

    con.close()


if __name__ == "__main__":
    main()
