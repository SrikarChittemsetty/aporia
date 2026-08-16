"""Bridge the gap between how people phrase claims and how philosophers wrote.

The retrieval eval turned up a specific, repeatable failure: dense retrieval
matches vocabulary and imagery, not the conclusion an argument reaches.

    "most people mistake shadows for reality"          -> no Plato
    "prisoners chained in a cave see only shadows"      -> Plato, immediately
    "morality is an invention of the weak"              -> no Nietzsche
    "master morality and slave morality"                -> Nietzsche, immediately

Both arguments are in the corpus. The user states the *upshot* in modern abstract
terms; Plato argues it by telling a story about a den and firelight and never
states the moral, and Nietzsche coins his own vocabulary rather than the
paraphrase he is remembered by. The embedding has no way to know these are the
same idea.

This closes the gap from the query side. Before searching, ask the model to write
a short passage *as a philosopher arguing the claim would have written it*, embed
that, and search with a blend of the two vectors. The hypothetical passage is
wrong about who said it and often wrong about the details — that does not matter,
because it is never shown to anyone. It exists only to land the query vector in
the region of the space where the real passages live. (This is the HyDE idea:
a hypothetical document is a better search key than the question.)

Same operational shape as the stance layer: one model call per unique claim,
cached in SQLite for ever, and retrieval degrades to plain vector search when no
backend is reachable rather than failing.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, QUERY_PREFIX
from api.stance import (
    _classify_via_cli_text,
    _complete_via_sdk,
    _query_hash,
    _sdk_available,
)

# How far to move the query toward the hypothetical passage. 0.0 is plain
# retrieval, 1.0 ignores the user's own wording entirely.
#
# 0.3 is not a taste judgement. On the held-out query set, 0.3 scores 20/21 and
# every stronger blend scores 19/21 — the same as no expansion at all. The user's
# own wording carries most of the signal; the hypothetical is a nudge toward the
# right region, and turning it into a shove throws away what the user actually
# asked for. (On the development set 0.3, 0.5 and 0.7 all score 41/41, which is
# exactly the kind of agreement that makes a development set look more decisive
# than it is.)
DEFAULT_ALPHA = 0.3

PROMPT_TEMPLATE = """Write a short passage (80-120 words) from a work of classical or early-modern philosophy that argues for this claim:

"{claim}"

Write it as the philosopher himself would have written it — his vocabulary, his imagery, his sentence rhythm, the concrete examples and analogies he would reach for rather than the abstract modern paraphrase. Do not name any philosopher, do not mention the claim in the words given above, and do not write a summary or commentary. Write only the passage itself, as if excerpted from the book."""


def _ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS query_expansions ("
        " query_hash TEXT PRIMARY KEY, claim TEXT NOT NULL, passage TEXT NOT NULL)"
    )


def cached(claim: str) -> str | None:
    con = sqlite3.connect(DB_PATH)
    _ensure_table(con)
    row = con.execute(
        "SELECT passage FROM query_expansions WHERE query_hash = ?",
        (_query_hash(claim),),
    ).fetchone()
    con.close()
    return row[0] if row else None


def store(claim: str, passage: str) -> None:
    con = sqlite3.connect(DB_PATH)
    _ensure_table(con)
    con.execute(
        "INSERT OR REPLACE INTO query_expansions (query_hash, claim, passage) VALUES (?,?,?)",
        (_query_hash(claim), claim, passage.strip()),
    )
    con.commit()
    con.close()


def expand(claim: str) -> tuple[str | None, str | None]:
    """Return (hypothetical_passage, error). Cache first, then the LLM."""
    hit = cached(claim)
    if hit:
        return hit, None

    prompt = PROMPT_TEMPLATE.format(claim=claim)
    try:
        raw = _complete_via_sdk(prompt) if _sdk_available() else _classify_via_cli_text(prompt)
    except Exception as e:  # noqa: BLE001 — no backend is a normal state here
        return None, f"expansion backend unavailable: {e}"

    passage = raw.strip()
    if not passage:
        return None, "expansion backend returned nothing"
    store(claim, passage)
    return passage, None


def search_vector(model, query: str, *, alpha: float = DEFAULT_ALPHA):
    """The vector to actually search with. Returns (vector, expansion, error).

    One place, so the live API and the static-site exporter cannot drift into
    retrieving differently. Falls back to the plain query vector whenever the
    expansion is unavailable — a search that returns slightly worse results
    beats a search that returns an error.
    """
    import numpy as np

    q_vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    if alpha <= 0.0:
        return np.asarray(q_vec, dtype=np.float32), None, None

    passage, error = expand(query)
    if passage is None:
        return np.asarray(q_vec, dtype=np.float32), None, error

    # The hypothetical is a passage, not a query, so it gets no query prefix —
    # the bge models are trained with that asymmetry.
    h_vec = model.encode(passage, normalize_embeddings=True)
    return blend(q_vec, h_vec, alpha), passage, None


def blend(query_vec, hyde_vec, alpha: float = DEFAULT_ALPHA):
    """Move the query vector toward the hypothetical passage and re-normalise.

    Both inputs are already unit vectors, so this is a point on the arc between
    them — `alpha` is how far along. Re-normalising matters because cosine
    similarity is what the index scores on.
    """
    import numpy as np

    v = (1.0 - alpha) * np.asarray(query_vec, dtype=np.float32) + alpha * np.asarray(
        hyde_vec, dtype=np.float32
    )
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else np.asarray(query_vec, dtype=np.float32)
