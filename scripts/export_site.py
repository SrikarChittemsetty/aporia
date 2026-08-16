"""Export Aporia as a static, zero-install demo site.

The live app needs a Python environment, a 28 MB index and an LLM backend. That
is a fine thing to ask of an engineer and an impossible thing to ask of anyone
else, so this script freezes a set of curated claims into a page that runs
entirely in the browser.

Nothing here fakes the pipeline. Retrieval is the real thing — the same
bge-small embeddings through the same from-scratch HNSW index that serves the
live app — and the stance labels come from the same Claude prompt the live app
sends, written into the same SQLite cache the live app reads. The static site is
a snapshot of real output, not a mock-up of it.

Three stages, because the classification step needs an LLM backend that may not
be reachable from this machine:

    python scripts/export_site.py retrieve    # real search -> retrieval.json + prompts
    python scripts/export_site.py ingest      # labels/*.json -> the stance cache
    python scripts/export_site.py build       # cache + retrieval -> docs/data.js

`retrieve` writes one classification prompt per claim, verbatim from
api/stance.PROMPT_TEMPLATE. Answer them with any Claude backend (the API, the
CLI, or a session) and drop the JSON replies in data/demo/labels/. If the live
backend is reachable, api.stance.classify does this by itself and `ingest` has
nothing left to do.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import expand, stance
from config import DB_PATH, EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, ROOT, TOP_K

DEMO_DIR = ROOT / "data" / "demo"
PROMPT_DIR = DEMO_DIR / "prompts"
LABEL_DIR = DEMO_DIR / "labels"
RETRIEVAL_PATH = DEMO_DIR / "retrieval.json"
DOCS_DIR = ROOT / "docs"

# The claims the static demo ships with. Chosen to span the corpus — each one
# has genuine defenders and genuine attackers among the 13 works, which is the
# only way a FOR/AGAINST split means anything.
DEMO_CLAIMS: list[dict] = [
    {"claim": "Humans have free will", "topic": "free will",
     "blurb": "The oldest fight in the corpus: Spinoza and Nietzsche against, James and Kant for."},
    {"claim": "God exists", "topic": "the existence of god",
     "blurb": "Hume's Dialogues stage both sides of the design argument in one book."},
    {"claim": "The design of the universe proves an intelligent creator", "topic": "the argument from design",
     "blurb": "Hume wrote both sides of this one himself, as two characters in one dialogue."},
    {"claim": "Morality is objective", "topic": "morality",
     "blurb": "Kant's categorical imperative meets Nietzsche's genealogy of morals."},
    {"claim": "Certain knowledge is possible", "topic": "knowledge",
     "blurb": "Descartes builds certainty from doubt; Hume takes it apart again."},
    {"claim": "Humans have natural rights that governments must respect", "topic": "natural rights",
     "blurb": "Locke and Paine found the modern state on a claim Hume never accepted."},
    {"claim": "Happiness is the highest good", "topic": "happiness",
     "blurb": "Mill's utilitarianism against everyone who thought duty came first."},
    {"claim": "Individual liberty may only be limited to prevent harm to others", "topic": "liberty",
     "blurb": "Mill's harm principle, and the objections it has always attracted."},
    {"claim": "Justice is objectively real, not mere convention", "topic": "justice",
     "blurb": "Plato's Republic exists because someone argued justice was a racket."},
    {"claim": "It is rational to believe something without sufficient evidence", "topic": "faith and evidence",
     "blurb": "James says yes and Hume says never — the fight over what belief owes to proof."},
    {"claim": "The self persists as one thing over time", "topic": "personal identity",
     "blurb": "Descartes' thinking thing versus Hume's bundle of perceptions."},
    {"claim": "Good and evil are objective, not human inventions", "topic": "good and evil",
     "blurb": "Beyond Good and Evil is a book-length answer to this one."},
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


# --- stage 1: real retrieval --------------------------------------------------


def stage_retrieve(k: int) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from index.vector_index import load_index

    print(f"loading {EMBED_MODEL} and the {INDEX_DIR.name} index…")
    model = SentenceTransformer(EMBED_MODEL)
    index = load_index(INDEX_DIR)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    out = []
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    missing_expansions = []
    for entry in DEMO_CLAIMS:
        claim = entry["claim"]
        # Same retrieval path the live app uses, so the frozen demo cannot drift
        # from what a local run would produce.
        vector, hypothetical, expand_error = expand.search_vector(model, claim)
        if hypothetical is None:
            missing_expansions.append(claim)
        ids, scores = index.search(np.asarray(vector, dtype=np.float32), k=k)
        score_by_id = dict(zip(ids, scores))

        placeholders = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT id, author, work, citation_path, text FROM chunks WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        passages = [
            {**by_id[i], "similarity": round(float(score_by_id[i]), 4)}
            for i in ids
            if i in by_id
        ]

        out.append({**entry, "passages": passages})

        # Write the classification prompt exactly as the live pipeline would send it.
        prompt = stance.PROMPT_TEMPLATE.format(
            query=claim, passages=stance._render_passages(passages)
        )
        (PROMPT_DIR / f"{slug(claim)}.txt").write_text(prompt)
        print(f"  {claim[:52]:<54} {len(passages)} passages")

    con.close()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_PATH.write_text(json.dumps(out, indent=2) + "\n")

    if missing_expansions:
        print(f"\n! {len(missing_expansions)} claim(s) have no cached query expansion; "
              f"they used plain retrieval:")
        for c in missing_expansions:
            print(f"    {c}")

    missing = [c for c in out if not _cached_ids(c["claim"])]
    print(f"\nwrote {RETRIEVAL_PATH.relative_to(ROOT)}")
    print(f"prompts in {PROMPT_DIR.relative_to(ROOT)} — {len(missing)} claim(s) still unclassified")


def _cached_ids(claim: str) -> set[int]:
    con = sqlite3.connect(DB_PATH)
    qh = stance._query_hash(claim)
    ids = {r[0] for r in con.execute(
        "SELECT chunk_id FROM stance_cache WHERE query_hash = ?", (qh,)
    )}
    con.close()
    return ids


# --- stage 2: fold labels into the cache the app already reads ----------------


def stage_ingest() -> None:
    data = json.loads(RETRIEVAL_PATH.read_text())
    con = sqlite3.connect(DB_PATH)
    total = 0

    for entry in data:
        claim = entry["claim"]
        path = LABEL_DIR / f"{slug(claim)}.json"
        if not path.exists():
            print(f"  (no labels yet) {claim}")
            continue

        raw = json.loads(path.read_text())
        if isinstance(raw, dict):  # tolerate {"labels": [...]}
            raw = raw.get("labels", [])
        by_id = {int(r["id"]): r for r in raw if "id" in r}

        valid_ids = {p["id"] for p in entry["passages"]}
        qh = stance._query_hash(claim)
        written = 0
        for pid in valid_ids:
            r = by_id.get(pid)
            if not r:
                continue
            s = r.get("stance", "nuance")
            if s not in stance.VALID_STANCES:
                s = "nuance"
            con.execute(
                "INSERT OR REPLACE INTO stance_cache (query_hash, chunk_id, stance, move, confidence)"
                " VALUES (?,?,?,?,?)",
                (qh, pid, s, str(r.get("move", ""))[:400], float(r.get("confidence", 0.0))),
            )
            written += 1

        unknown = set(by_id) - valid_ids
        if unknown:
            print(f"  ! {claim}: {len(unknown)} label id(s) not in the retrieved set, ignored")
        print(f"  {claim[:52]:<54} {written}/{len(valid_ids)} labelled")
        total += written

    con.commit()
    con.close()
    print(f"\n{total} labels written to the stance cache at {DB_PATH.relative_to(ROOT)}")


# --- stage 3: freeze it into a static page -----------------------------------


def stage_build() -> None:
    data = json.loads(RETRIEVAL_PATH.read_text())
    con = sqlite3.connect(DB_PATH)

    claims = []
    counts = {"for": 0, "against": 0, "nuance": 0}
    unlabelled = 0

    for entry in data:
        claim = entry["claim"]
        qh = stance._query_hash(claim)
        cached = {
            cid: {"stance": s, "move": m, "confidence": c}
            for cid, s, m, c in con.execute(
                "SELECT chunk_id, stance, move, confidence FROM stance_cache WHERE query_hash = ?",
                (qh,),
            )
        }

        grouped: dict[str, list] = {"for": [], "against": [], "nuance": []}
        for p in entry["passages"]:
            label = cached.get(p["id"])
            if label is None:
                unlabelled += 1
                label = {"stance": "nuance", "move": "", "confidence": 0.0}
            grouped[label["stance"]].append(
                {
                    "id": p["id"],
                    "author": p["author"],
                    "work": p["work"],
                    "citation": p["citation_path"],
                    "text": p["text"],
                    "move": label["move"],
                    "confidence": round(float(label["confidence"]), 2),
                    "similarity": p["similarity"],
                }
            )
            counts[label["stance"]] += 1

        claims.append(
            {
                "claim": claim,
                "topic": entry["topic"],
                "blurb": entry["blurb"],
                "for": grouped["for"],
                "against": grouped["against"],
                "nuance": grouped["nuance"],
            }
        )

    corpus = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    works = con.execute("SELECT count(DISTINCT work) FROM chunks").fetchone()[0]
    authors = con.execute("SELECT count(DISTINCT author) FROM chunks").fetchone()[0]
    con.close()

    payload = {
        "claims": claims,
        "stats": {
            "passages": corpus,
            "works": works,
            "authors": authors,
            "claims": len(claims),
            "classified": sum(counts.values()),
            "for": counts["for"],
            "against": counts["against"],
            "nuance": counts["nuance"],
        },
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "data.js").write_text(
        "// Generated by scripts/export_site.py.\n"
        "// Retrieval is real: bge-small embeddings through the from-scratch HNSW\n"
        "// index. Stance labels come from the pipeline's own Claude prompt.\n"
        f"window.APORIA_DATA = {json.dumps(payload, ensure_ascii=False, indent=1)};\n"
    )

    size_kb = (DOCS_DIR / "data.js").stat().st_size / 1024
    print(f"wrote {(DOCS_DIR / 'data.js').relative_to(ROOT)} ({size_kb:.0f} KB)")
    print(f"  {len(claims)} claims, {sum(counts.values())} classified passages")
    print(f"  for={counts['for']}  against={counts['against']}  nuance={counts['nuance']}")
    if unlabelled:
        print(f"  ! {unlabelled} passage(s) had no cached label and defaulted to nuance")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["retrieve", "ingest", "build"])
    ap.add_argument("-k", type=int, default=TOP_K, help="passages per claim")
    args = ap.parse_args()

    {"retrieve": lambda: stage_retrieve(args.k), "ingest": stage_ingest, "build": stage_build}[
        args.stage
    ]()


if __name__ == "__main__":
    main()
