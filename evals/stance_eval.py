"""Measure live stance-layer agreement against the hand-labeled gold set.

Usage: python -m evals.stance_eval
Requires a working LLM backend (ANTHROPIC_API_KEY or an authenticated
`claude` CLI) — this deliberately bypasses the stance cache to test the live
classifier, and reports per-stance agreement plus a confusion matrix.
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH
from api import stance

GOLD = Path(__file__).parent / "gold_stances.json"


def main() -> None:
    labels = json.loads(GOLD.read_text())
    by_claim = defaultdict(list)
    for row in labels:
        by_claim[row["claim"]].append(row)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    agree = total = 0
    confusion = Counter()
    for claim, rows in by_claim.items():
        ids = [r["chunk_id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        chunks = {c["id"]: dict(c) for c in con.execute(
            f"SELECT id, author, work, citation_path, text FROM chunks WHERE id IN ({placeholders})", ids)}
        passages = [chunks[i] for i in ids if i in chunks]

        prompt = stance.PROMPT_TEMPLATE.format(
            query=claim, passages=stance._render_passages(passages))
        raw = (stance._classify_via_sdk(prompt) if stance._sdk_available()
               else stance._classify_via_cli(prompt))
        predicted = {int(r["id"]): r.get("stance", "nuance") for r in raw if "id" in r}

        for r in rows:
            pred = predicted.get(r["chunk_id"])
            if pred is None:
                continue
            total += 1
            agree += pred == r["stance"]
            confusion[(r["stance"], pred)] += 1
        print(f"{claim[:60]:62} done")
    con.close()

    print(f"\nagreement: {agree}/{total} = {agree/total:.1%}")
    print("\nconfusion (gold -> predicted):")
    for stances in ("for", "against", "nuance"):
        row = "  ".join(f"{p}:{confusion[(stances, p)]:3d}" for p in ("for", "against", "nuance"))
        print(f"  gold {stances:8} -> {row}")


if __name__ == "__main__":
    main()
