"""Sanity-check retrieval + stance on hand-picked free-will queries.

Usage: python -m evals.sanity          (retrieval only, no LLM calls)
       python -m evals.sanity --stance (also runs stance classification)

Expectations are loose on purpose: each query lists authors we'd expect to
see somewhere in the top-K. Prints hit/miss per query and a summary.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, TOP_K
from index.vector_index import load_index

QUERIES = [
    ("free will is an illusion", {"Baruch Spinoza"}),
    ("liberty is compatible with necessity", {"David Hume"}),
    ("all human actions are determined by prior causes", {"Baruch Spinoza", "David Hume"}),
    ("determinism makes moral responsibility impossible", {"William James"}),
    ("chance and novelty are real features of the world", {"William James"}),
    ("freedom is a postulate of morality", {"Immanuel Kant"}),
    ("men believe themselves free because they are ignorant of the causes of their actions", {"Baruch Spinoza"}),
    ("the will is determined by motives", {"David Hume", "Baruch Spinoza"}),
    ("moral regret only makes sense if we could have done otherwise", {"William James"}),
    ("practical reason requires assuming freedom", {"Immanuel Kant"}),
]


def main() -> None:
    run_stance = "--stance" in sys.argv

    import sqlite3

    from sentence_transformers import SentenceTransformer

    from config import DB_PATH

    model = SentenceTransformer(EMBED_MODEL)
    index = load_index(INDEX_DIR)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    hits = 0
    for query, expected_authors in QUERIES:
        vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        ids, _ = index.search(np.asarray(vec, dtype=np.float32), k=TOP_K)
        placeholders = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT id, author, work, citation_path, text FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        authors = {r["author"] for r in rows}
        ok = bool(expected_authors & authors)
        hits += ok
        print(f"[{'HIT ' if ok else 'MISS'}] {query!r}")
        print(f"        retrieved authors: {sorted(authors)}")
        if run_stance:
            from api import stance

            passages = [dict(r) for r in rows]
            s, err = stance.classify(query, passages)
            counts = {"for": 0, "against": 0, "nuance": 0}
            for v in s.values():
                counts[v["stance"]] += 1
            print(f"        stance split: {counts}" + (f"  (ERROR: {err})" if err else ""))
    con.close()
    print(f"\nretrieval: {hits}/{len(QUERIES)} queries surfaced an expected author in top-{TOP_K}")


if __name__ == "__main__":
    main()
