"""Build a blind labelling sheet, so the classifier can be scored against a human.

Everything currently measured about the stance layer is self-referential: labels
come from a model, and `evals/stability.py` checks whether the model agrees with
itself. That is worth knowing and it is not accuracy. Accuracy needs someone who
knows the texts to read the passages cold and say what they argue.

This writes a single self-contained HTML file that shows one passage at a time
with the claim above it and the model's own answer hidden. You press for /
against / nuance / skip, and at the end you copy out a JSON array. Feed that to
evals/score_gold.py and you have a real number.

    python -m evals.make_labeling_sheet --n 60
    open evals/labeling_sheet.html          # label them
    # paste the result into evals/human_labels.json
    python -m evals.score_gold

Sampling is stratified across claims and shuffled with a fixed seed, so the sheet
is reproducible and no claim dominates. The model's labels are not in the file at
all — not hidden in an attribute, not in a comment — so there is nothing to
anchor on even accidentally.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT

RETRIEVAL = ROOT / "data" / "demo" / "retrieval.json"
OUT = Path(__file__).parent / "labeling_sheet.html"

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aporia — blind stance labelling</title>
<style>
 :root {{ --bg:#faf7f2; --ink:#22201c; --muted:#7a746a; --line:#e4ddd2; --accent:#b4541f;
          --for:#1e6f50; --against:#9c3524; --nuance:#6b5ca5; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:17px/1.6 Georgia,serif; }}
 .wrap {{ max-width:760px; margin:0 auto; padding:2rem 1.5rem 5rem; }}
 .bar {{ height:6px; background:var(--line); border-radius:4px; overflow:hidden; margin-bottom:1.5rem; }}
 .bar i {{ display:block; height:100%; background:var(--accent); transition:width .2s; }}
 .meta {{ font:13px -apple-system,sans-serif; color:var(--muted); letter-spacing:.06em;
          text-transform:uppercase; margin-bottom:.5rem; }}
 .claim {{ font-size:1.35rem; margin:0 0 1.5rem; }}
 .claim b {{ color:var(--accent); }}
 blockquote {{ margin:0 0 1.5rem; padding:1.2rem 1.4rem; background:#fffdfa;
               border:1px solid var(--line); border-radius:10px; }}
 .cite {{ font-size:.9rem; color:var(--muted); margin-top:.8rem; }}
 .btns {{ display:flex; gap:.6rem; flex-wrap:wrap; }}
 button {{ font:16px Georgia,serif; padding:.7rem 1.3rem; border:1px solid var(--line);
           border-radius:9px; background:#fffdfa; cursor:pointer; color:var(--ink); }}
 button:hover {{ border-color:var(--accent); }}
 button.f {{ color:var(--for); }} button.a {{ color:var(--against); }} button.n {{ color:var(--nuance); }}
 kbd {{ font:12px ui-monospace,monospace; background:var(--line); padding:.1em .4em; border-radius:4px; }}
 #done {{ display:none; }}
 textarea {{ width:100%; height:340px; font:12px ui-monospace,monospace; padding:1rem;
             border:1px solid var(--line); border-radius:9px; background:#fffdfa; }}
 .hint {{ font-size:.9rem; color:var(--muted); margin-top:1.2rem; }}
</style></head><body><div class="wrap">
<div id="card">
  <div class="bar"><i id="prog" style="width:0%"></i></div>
  <p class="meta" id="count"></p>
  <p class="claim">Does this passage argue <b id="claim"></b>?</p>
  <blockquote><span id="text"></span><div class="cite" id="cite"></div></blockquote>
  <div class="btns">
    <button class="f" data-s="for">Argues FOR <kbd>f</kbd></button>
    <button class="a" data-s="against">Argues AGAINST <kbd>a</kbd></button>
    <button class="n" data-s="nuance">Reframes / neither <kbd>n</kbd></button>
    <button data-s="skip">Skip <kbd>s</kbd></button>
  </div>
  <p class="hint">Judge only what this passage says. Ignore what you know the author
     argued elsewhere. If it discusses the topic without taking a side, that is
     <em>reframes / neither</em>.</p>
</div>
<div id="done">
  <h2>Done — {n} passages</h2>
  <p>Copy this into <code>evals/human_labels.json</code>, then run
     <code>python -m evals.score_gold</code>.</p>
  <textarea id="out" readonly></textarea>
</div>
</div>
<script>
const ITEMS = {items};
let i = 0; const answers = [];
const $ = id => document.getElementById(id);
function show() {{
  if (i >= ITEMS.length) {{
    $("card").style.display = "none"; $("done").style.display = "block";
    $("out").value = JSON.stringify(answers, null, 1); $("out").select(); return;
  }}
  const it = ITEMS[i];
  $("count").textContent = `passage ${{i + 1}} of ${{ITEMS.length}}`;
  $("prog").style.width = (100 * i / ITEMS.length) + "%";
  $("claim").textContent = "\\u201c" + it.claim + "\\u201d";
  $("text").textContent = it.text;
  $("cite").textContent = it.author + ", " + it.work + " — " + it.citation;
  window.scrollTo(0, 0);
}}
function answer(s) {{
  if (s !== "skip") answers.push({{ claim: ITEMS[i].claim, chunk_id: ITEMS[i].id, stance: s }});
  i++; show();
}}
document.querySelectorAll("button").forEach(b =>
  b.addEventListener("click", () => answer(b.dataset.s)));
document.addEventListener("keydown", e => {{
  const map = {{ f: "for", a: "against", n: "nuance", s: "skip" }};
  if (map[e.key]) answer(map[e.key]);
}});
show();
</script></body></html>
"""


def sample_passages(n: int, seed: int) -> list[dict]:
    """The stratified sample both blind evals draw from.

    Shared (rather than duplicated) so the human sheet and the cross-prompt
    judge in evals/judge_eval.py score the *same* passages — same n, same seed,
    same items — and their numbers stay comparable.
    """
    data = json.loads(RETRIEVAL.read_text())
    rng = random.Random(seed)

    # Stratified: take an equal slice from each claim, then top up at random.
    per_claim = max(1, n // len(data))
    picked, pool = [], []
    for entry in data:
        passages = list(entry["passages"])
        rng.shuffle(passages)
        for p in passages[:per_claim]:
            picked.append((entry["claim"], p))
        for p in passages[per_claim:]:
            pool.append((entry["claim"], p))

    rng.shuffle(pool)
    picked.extend(pool[: max(0, n - len(picked))])
    rng.shuffle(picked)

    return [
        {
            "claim": claim,
            "id": p["id"],
            "text": p["text"],
            "author": p["author"],
            "work": p["work"],
            "citation": p["citation_path"],
        }
        for claim, p in picked[:n]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="how many passages to label")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    items = sample_passages(args.n, args.seed)

    OUT.write_text(
        TEMPLATE.format(items=json.dumps(items, ensure_ascii=False), n=len(items))
    )
    claims = len({it["claim"] for it in items})
    print(f"wrote {OUT} — {len(items)} passages across {claims} claims (seed {args.seed})")
    print("open it, label them, and save the JSON to evals/human_labels.json")


if __name__ == "__main__":
    main()
