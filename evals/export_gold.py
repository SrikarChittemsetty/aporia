"""Export the hand-labeled stance cache as a gold set for stance_eval.py.

Usage: python -m evals.export_gold
Writes evals/gold_stances.json: [{claim, chunk_id, stance}, ...]
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH
from api.stance import _query_hash

# The claims whose cached classifications were hand-reviewed.
GOLD_CLAIMS = [
    "free will is an illusion",
    "liberty is compatible with necessity",
    "determinism makes morality meaningless",
    "Humans have free will",
    "God exists",
    "The existence of evil is incompatible with a good God",
]

OUT = Path(__file__).parent / "gold_stances.json"


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    rows = []
    for claim in GOLD_CLAIMS:
        qh = _query_hash(claim)
        for cid, stance in con.execute(
            "SELECT chunk_id, stance FROM stance_cache WHERE query_hash = ?", (qh,)
        ):
            # Skip labels for chunks that no longer exist (corpus rebuilds).
            if con.execute("SELECT 1 FROM chunks WHERE id = ?", (cid,)).fetchone():
                rows.append({"claim": claim, "chunk_id": cid, "stance": stance})
    con.close()
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} gold labels -> {OUT}")


if __name__ == "__main__":
    main()
