"""Score the stance classifier against human labels — the real accuracy number.

Reads evals/human_labels.json (produced by the blind sheet from
evals/make_labeling_sheet.py) and compares it to what the classifier put in the
stance cache for the same (claim, passage) pairs.

    python -m evals.score_gold

Unlike evals/stability.py, this is an accuracy measurement: the two label sets
come from different judges, one of whom is a person who read the passage without
seeing the model's answer.

Reports overall accuracy, a confusion matrix, per-stance precision and recall,
and Cohen's kappa — which is the number to quote, because raw agreement flatters
any classifier on a skewed label distribution. Kappa asks how much better than
chance the agreement is.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import stance
from config import DB_PATH

HUMAN = Path(__file__).parent / "human_labels.json"
STANCES = ("for", "against", "nuance")


def cohens_kappa(confusion: Counter[tuple[str, str]], n: int) -> float:
    """Agreement corrected for the agreement you'd get by guessing."""
    observed = sum(confusion[(s, s)] for s in STANCES) / n
    expected = 0.0
    for s in STANCES:
        human_rate = sum(confusion[(s, m)] for m in STANCES) / n
        model_rate = sum(confusion[(h, s)] for h in STANCES) / n
        expected += human_rate * model_rate
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    if not HUMAN.exists():
        print(f"no {HUMAN.name} yet.\n")
        print("  python -m evals.make_labeling_sheet --n 60")
        print("  open evals/labeling_sheet.html   # label them, copy the JSON out")
        print(f"  save it to {HUMAN}")
        sys.exit(1)

    human = json.loads(HUMAN.read_text())
    con = sqlite3.connect(DB_PATH)

    confusion: Counter[tuple[str, str]] = Counter()  # (human, model)
    missing = 0

    for row in human:
        qh = stance._query_hash(row["claim"])
        got = con.execute(
            "SELECT stance FROM stance_cache WHERE query_hash = ? AND chunk_id = ?",
            (qh, row["chunk_id"]),
        ).fetchone()
        if got is None:
            missing += 1
            continue
        confusion[(row["stance"], got[0])] += 1

    con.close()

    n = sum(confusion.values())
    if not n:
        print("no overlap between the human labels and the stance cache")
        sys.exit(1)

    correct = sum(confusion[(s, s)] for s in STANCES)
    print(f"labelled by hand:     {len(human)}")
    if missing:
        print(f"not in the cache:     {missing} (skipped)")
    print(f"compared:             {n}")
    print(f"accuracy:             {correct}/{n} = {correct / n:.1%}")
    print(f"Cohen's kappa:        {cohens_kappa(confusion, n):.3f}")

    print("\nconfusion (human → model):")
    print("  " + " " * 12 + "".join(f"{s:>10}" for s in STANCES))
    for h in STANCES:
        print(f"  {h:<12}" + "".join(f"{confusion[(h, m)]:>10}" for m in STANCES))

    print("\nper stance:")
    for s in STANCES:
        tp = confusion[(s, s)]
        model_said = sum(confusion[(h, s)] for h in STANCES)
        human_said = sum(confusion[(s, m)] for m in STANCES)
        prec = tp / model_said if model_said else 0.0
        rec = tp / human_said if human_said else 0.0
        print(f"  {s:<10} precision {prec:5.1%}   recall {rec:5.1%}   (n={human_said})")

    reversals = confusion[("for", "against")] + confusion[("against", "for")]
    print(f"\noutright reversals (human for ↔ model against): {reversals}")


if __name__ == "__main__":
    main()
