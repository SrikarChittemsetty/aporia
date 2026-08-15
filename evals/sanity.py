"""Does retrieval surface the philosopher who actually holds the position?

Usage: python -m evals.sanity          (retrieval only, no LLM calls)
       python -m evals.sanity --stance (also runs stance classification)
       python -m evals.sanity --verbose (show what came back for every query)

The eval used to be ten queries, all about free will — a leftover from when the
corpus was four books on that one debate. Against a corpus that now spans God,
morality, rights, knowledge and justice, a score measured on that slice said very
little: it could sit at 9/10 while retrieval failed completely on eight of the
thirteen works.

So this set covers every work in the corpus, three or four queries each. The
queries are written the way someone would actually type them — a claim or a
question, not a phrase lifted from the text — because retrieval that only works
when you already know the wording is not retrieval.

Each query names the author(s) who genuinely argue the position somewhere in the
corpus; a hit means at least one of them appears in the top-K. That is a loose
bar on purpose. It asks "did the right thinker show up at all", not "was the
ranking ideal", and it is the strongest claim a set this size can support.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, TOP_K
from index.vector_index import load_index

SPINOZA = "Baruch Spinoza"
HUME = "David Hume"
NIETZSCHE = "Friedrich Nietzsche"
KANT = "Immanuel Kant"
LOCKE = "John Locke"
MILL = "John Stuart Mill"
PLATO = "Plato"
DESCARTES = "René Descartes"
PAINE = "Thomas Paine"
JAMES = "William James"

# (query, authors who actually argue this in the corpus, which debate it belongs to)
QUERIES: list[tuple[str, set[str], str]] = [
    # --- free will and determinism -------------------------------------------
    ("free will is an illusion", {SPINOZA}, "free will"),
    ("liberty is compatible with necessity", {HUME}, "free will"),
    ("all human actions are determined by prior causes", {SPINOZA, HUME}, "free will"),
    ("determinism makes moral responsibility impossible", {JAMES}, "free will"),
    ("chance and novelty are real features of the world", {JAMES}, "free will"),
    ("freedom is a postulate of morality", {KANT}, "free will"),
    ("we feel free only because we do not know what causes us to act", {SPINOZA}, "free will"),
    ("moral regret only makes sense if we could have done otherwise", {JAMES}, "free will"),

    # --- God, design and religion --------------------------------------------
    ("the order of nature proves an intelligent designer", {HUME}, "god"),
    ("the universe resembles a machine, so it must have had a maker", {HUME}, "god"),
    ("we cannot infer an infinite creator from a finite world", {HUME}, "god"),
    ("belief in God is a postulate of practical reason", {KANT}, "god"),
    ("it is reasonable to believe without conclusive evidence", {JAMES}, "god"),
    ("testimony can never establish that a miracle occurred", {HUME}, "god"),

    # --- knowledge and doubt --------------------------------------------------
    ("I can doubt everything except that I am thinking", {DESCARTES}, "knowledge"),
    ("we should reject any belief we can find the slightest reason to doubt",
     {DESCARTES}, "knowledge"),
    ("we have no rational justification for expecting the future to resemble the past",
     {HUME}, "knowledge"),
    ("causation is a habit of the mind, not something we observe", {HUME}, "knowledge"),
    ("the senses deceive us, so reason must be the foundation of knowledge",
     {DESCARTES, PLATO}, "knowledge"),

    # --- morality -------------------------------------------------------------
    ("act only on a rule you could will everyone to follow", {KANT}, "morality"),
    ("people must be treated as ends in themselves, never merely as means",
     {KANT}, "morality"),
    ("an action has moral worth only when done from duty", {KANT}, "morality"),
    ("morality is an invention of the weak to constrain the strong", {NIETZSCHE}, "morality"),
    ("good and evil are historical inventions rather than eternal facts",
     {NIETZSCHE}, "morality"),
    ("philosophers have smuggled their prejudices in as reasoning", {NIETZSCHE}, "morality"),
    ("the right action is the one producing the greatest happiness", {MILL}, "morality"),
    ("some pleasures are higher in kind than others", {MILL}, "morality"),

    # --- politics, rights and liberty ----------------------------------------
    ("people are born free and equal in a state of nature", {LOCKE}, "politics"),
    ("mixing your labour with something makes it your property", {LOCKE}, "politics"),
    ("government is legitimate only with the consent of the governed",
     {LOCKE, PAINE}, "politics"),
    ("a people may overthrow a government that betrays its trust", {LOCKE, PAINE}, "politics"),
    ("hereditary rule is an absurdity no generation can impose on the next",
     {PAINE}, "politics"),
    ("rights belong to men as men, not as gifts from a sovereign", {PAINE}, "politics"),
    ("power may only be used against someone to prevent harm to others", {MILL}, "politics"),
    ("silencing an opinion robs those who disagree with it", {MILL}, "politics"),
    ("the majority can tyrannise as effectively as any despot", {MILL}, "politics"),

    # --- justice and the soul -------------------------------------------------
    ("justice is nothing but the interest of the stronger", {PLATO}, "justice"),
    ("the state should be ruled by those who know the good", {PLATO}, "justice"),
    ("the just soul has its parts in their proper order", {PLATO}, "justice"),
    ("most people mistake shadows for reality", {PLATO}, "justice"),
    ("justice is a name for the rules that matter most to human well-being",
     {MILL}, "justice"),
]


def main() -> None:
    run_stance = "--stance" in sys.argv
    verbose = "--verbose" in sys.argv

    import sqlite3

    from sentence_transformers import SentenceTransformer

    from config import DB_PATH

    model = SentenceTransformer(EMBED_MODEL)
    index = load_index(INDEX_DIR)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    hits = 0
    misses: list[tuple[str, set[str], list[str]]] = []
    by_topic: dict[str, list[int]] = defaultdict(list)
    works_seen: set[str] = set()

    for query, expected_authors, topic in QUERIES:
        vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        ids, _ = index.search(np.asarray(vec, dtype=np.float32), k=TOP_K)
        placeholders = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT id, author, work, citation_path, text FROM chunks"
            f" WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        authors = {r["author"] for r in rows}
        works_seen.update(r["work"] for r in rows)

        ok = bool(expected_authors & authors)
        hits += ok
        by_topic[topic].append(int(ok))
        if not ok:
            misses.append((query, expected_authors, sorted(authors)))

        if verbose:
            print(f"[{'HIT ' if ok else 'MISS'}] {query!r}")
            print(f"        retrieved: {sorted(authors)}")

        if run_stance:
            from api import stance

            passages = [dict(r) for r in rows]
            s, err = stance.classify(query, passages)
            counts = {"for": 0, "against": 0, "nuance": 0}
            for v in s.values():
                counts[v["stance"]] += 1
            print(f"        stance split: {counts}" + (f"  (ERROR: {err})" if err else ""))

    total_works = con.execute("SELECT count(DISTINCT work) FROM chunks").fetchone()[0]
    con.close()

    print(f"\n{'=' * 64}")
    print(f"retrieval: {hits}/{len(QUERIES)} queries surfaced an expected author "
          f"in top-{TOP_K}  ({hits / len(QUERIES):.0%})")

    print("\nby debate:")
    for topic, results in sorted(by_topic.items()):
        print(f"  {topic:<12} {sum(results)}/{len(results)}")

    # A high score achieved by only ever returning the same two books would be a
    # bad score wearing a disguise, so report the spread as well.
    print(f"\ncorpus coverage: these queries pulled passages from "
          f"{len(works_seen)}/{total_works} works")

    if misses:
        print(f"\n{len(misses)} miss(es) — the honest part of the number:")
        for query, expected, got in misses:
            print(f"  {query!r}")
            print(f"     wanted {sorted(expected)}, got {got}")


if __name__ == "__main__":
    main()
