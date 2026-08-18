"""Cross-prompt judge: a second, independent LLM reading of the same passages.

What this is — and, stated first, what it is not. It is NOT a gold set and it
does NOT measure accuracy. Both judges here are language models, so their
agreement cannot certify that either is right; the human gold set
(make_labeling_sheet.py → human_labels.json → score_gold.py) remains the only
path to a real accuracy number, and remains open work.

What it measures instead is something self-consistency (stability.py) cannot:
whether the classifier's answers survive a change of *framing*. The production
classifier judges passages in a batch, as JSON, with ids, under one prompt
wording. The judge here reads one passage at a time, in plain language, with no
JSON, no ids, no batch — and no access to the classifier's answer. If the two
disagree on a passage, at least one of them is being steered by its prompt
rather than by the text. Run-to-run stability is necessary but cannot see this
failure mode, because reruns share the prompt.

    python -m evals.judge_eval --n 30

Writes evals/judge_results.json and prints agreement, Cohen's kappa, and the
confusion matrix (classifier × judge).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import stance
from config import DB_PATH
from evals.make_labeling_sheet import sample_passages
from evals.score_gold import STANCES, cohens_kappa

OUT = Path(__file__).parent / "judge_results.json"

# Deliberately unlike api/stance.py's prompt in every incidental way: single
# passage, second person, no JSON, no ids, no confidence, no "move". The one
# thing held constant is the *question*, because that is what's being measured.
JUDGE_PROMPT = """\
You are reading one passage of primary-source philosophy, cold.

The claim under discussion: "{claim}"

The passage, from {author}, {work}:

{text}

Does this passage argue FOR the claim, argue AGAINST the claim, or does it
reframe / qualify / sit genuinely between the two? Judge only what the passage
argues on the page — not what its author believed elsewhere.

Answer with exactly one word: for, against, or nuance."""


def ask_judge(item: dict) -> str:
    claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    prompt = JUDGE_PROMPT.format(
        claim=item["claim"], author=item["author"], work=item["work"], text=item["text"]
    )
    result = subprocess.run(
        [claude, "-p", prompt],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr.strip()[:200]}")
    # Take the last recognisable stance word: models sometimes preface with a
    # clause even when told not to, and the verdict lands at the end.
    words = re.findall(r"\b(for|against|nuance)\b", result.stdout.lower())
    return words[-1] if words else "unparseable"


def cached_label(con: sqlite3.Connection, claim: str, chunk_id: int) -> str | None:
    row = con.execute(
        "SELECT stance FROM stance_cache WHERE query_hash = ? AND chunk_id = ?",
        (stance._query_hash(claim), chunk_id),
    ).fetchone()
    return row[0] if row else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7, help="same default as the human sheet")
    ap.add_argument("--labels", type=Path, default=None, help=(
        "JSON file of pre-collected judge verdicts "
        "([{chunk_id, verdict}, ...]) instead of calling the claude CLI. "
        "Provenance (who produced them, how) is recorded in the output."
    ))
    ap.add_argument("--judge-desc", default="claude CLI, single-passage prompt",
                    help="recorded verbatim in judge_results.json as the judge's provenance")
    args = ap.parse_args()

    external: dict[int, str] = {}
    if args.labels:
        external = {int(r["chunk_id"]): r["verdict"] for r in json.loads(args.labels.read_text())}

    items = sample_passages(args.n, args.seed)
    con = sqlite3.connect(DB_PATH)

    rows, confusion = [], Counter()
    skipped_uncached = skipped_unparseable = 0

    for i, item in enumerate(items, 1):
        model = cached_label(con, item["claim"], item["id"])
        if model is None:
            skipped_uncached += 1
            continue

        judge = external.get(item["id"], "unparseable") if args.labels else ask_judge(item)
        print(f"  [{i:>2}/{len(items)}] classifier={model:<8} judge={judge:<12} "
              f"{item['author']}", file=sys.stderr)
        if judge not in STANCES:
            skipped_unparseable += 1
            continue

        confusion[(model, judge)] += 1
        rows.append({
            "claim": item["claim"], "chunk_id": item["id"],
            "author": item["author"], "work": item["work"],
            "classifier": model, "judge": judge,
        })

    con.close()

    n = sum(confusion.values())
    if not n:
        print("nothing to compare — is the stance cache populated?")
        sys.exit(1)

    agree = sum(confusion[(s, s)] for s in STANCES)
    kappa = cohens_kappa(confusion, n)

    OUT.write_text(json.dumps({
        "what_this_is": (
            "Agreement between the production stance classifier (batched JSON "
            "prompt) and an independently-prompted judge, both LLMs. This is a "
            "prompt-robustness check, NOT accuracy — no human labels are "
            "involved."
        ),
        "judge": args.judge_desc,
        "n": n, "agreement": agree / n, "kappa": kappa,
        "skipped_uncached": skipped_uncached,
        "skipped_unparseable": skipped_unparseable,
        "confusion_classifier_x_judge": {
            f"{a}->{b}": c for (a, b), c in sorted(confusion.items())
        },
        "disagreements": [r for r in rows if r["classifier"] != r["judge"]],
    }, indent=2, ensure_ascii=False))

    print(f"\ncompared:            {n}")
    print(f"agreement:           {agree}/{n} = {agree / n:.1%}")
    print(f"Cohen's kappa:       {kappa:.3f}")
    print("\nconfusion (classifier → judge):")
    print("  " + " " * 12 + "".join(f"{s:>10}" for s in STANCES))
    for a in STANCES:
        print(f"  {a:<12}" + "".join(f"{confusion[(a, b)]:>10}" for b in STANCES))
    print(f"\nwrote {OUT}")
    print("\nReminder: this is cross-prompt agreement between two LLM judges."
          "\nIt is not accuracy. The human gold set remains open work.")


if __name__ == "__main__":
    main()
