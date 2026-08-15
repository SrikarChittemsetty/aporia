"""How reproducible is the stance classifier?

A stance label is a judgement call, and the honest question about any judgement
call made by a model is whether it is stable — ask the same question twice, in
separate sessions with no memory of the first, and how often do you get the same
answer? A classifier that flips a third of its labels between runs is not one you
can build a search UI on, no matter how good any single run looks.

This compares an independent second pass against the labels already in the stance
cache and reports agreement plus a full confusion matrix.

    python -m evals.stability                      # compare pass 2 to the cache
    python -m evals.stability --prompts-out DIR    # write the prompts to run pass 2

What this is NOT: an accuracy measurement. Both passes come from the same model,
so agreement between them says the classifier is self-consistent, not that it is
right. Accuracy needs labels from a human who knows the texts — see
evals/gold_stances.json and the note in the README about what that file currently
is and is not.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import stance
from config import DB_PATH, ROOT

RETRIEVAL = ROOT / "data" / "demo" / "retrieval.json"
PASS2_DIR = ROOT / "data" / "demo" / "labels_pass2"
STANCES = ("for", "against", "nuance")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def write_prompts(out_dir: Path) -> None:
    data = json.loads(RETRIEVAL.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in data:
        prompt = stance.PROMPT_TEMPLATE.format(
            query=entry["claim"], passages=stance._render_passages(entry["passages"])
        )
        (out_dir / f"{slug(entry['claim'])}.txt").write_text(prompt)
    print(f"wrote {len(data)} prompts to {out_dir}")


def compare() -> None:
    data = json.loads(RETRIEVAL.read_text())
    con = sqlite3.connect(DB_PATH)

    agree = total = 0
    confusion: Counter[tuple[str, str]] = Counter()
    per_claim: dict[str, tuple[int, int]] = {}
    missing_files = []

    for entry in data:
        claim = entry["claim"]
        path = PASS2_DIR / f"{slug(claim)}.json"
        if not path.exists():
            missing_files.append(claim)
            continue

        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            raw = raw.get("labels", [])
        pass2 = {int(r["id"]): r.get("stance", "nuance") for r in raw if "id" in r}

        qh = stance._query_hash(claim)
        pass1 = dict(
            con.execute(
                "SELECT chunk_id, stance FROM stance_cache WHERE query_hash = ?", (qh,)
            )
        )

        c_agree = c_total = 0
        for pid, first in pass1.items():
            second = pass2.get(pid)
            if second is None:
                continue
            c_total += 1
            c_agree += first == second
            confusion[(first, second)] += 1
        per_claim[claim] = (c_agree, c_total)
        agree += c_agree
        total += c_total

    con.close()

    if not total:
        print("no overlapping labels found — run --prompts-out first, then answer the prompts")
        return

    print(f"{'claim':<58} agreement")
    for claim, (a, t) in sorted(per_claim.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        bar = "" if not t else f"{a}/{t}  {a / t:5.0%}"
        print(f"  {claim[:56]:<58}{bar}")

    print(f"\nrun-to-run agreement: {agree}/{total} = {agree / total:.1%}")

    print("\nconfusion (pass 1 → pass 2):")
    header = "  " + " " * 12 + "".join(f"{s:>10}" for s in STANCES)
    print(header)
    for first in STANCES:
        row = "".join(f"{confusion[(first, second)]:>10}" for second in STANCES)
        print(f"  {first:<12}{row}")

    # Which direction does disagreement run? Nearly always in and out of nuance,
    # which is the label a classifier reaches for when a passage is genuinely
    # equivocal — a much less alarming instability than for↔against flips.
    flips = confusion[("for", "against")] + confusion[("against", "for")]
    print(f"\noutright reversals (for ↔ against): {flips}")
    if missing_files:
        print(f"\nno pass-2 file for {len(missing_files)} claim(s): {', '.join(missing_files[:3])}…")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-out", type=Path, default=None)
    args = ap.parse_args()
    if args.prompts_out:
        write_prompts(args.prompts_out)
    else:
        compare()


if __name__ == "__main__":
    main()
