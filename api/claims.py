"""Resolve a user query into the claim that gets debated.

Claim-shaped queries ("free will is an illusion") pass through unchanged.
Bare topics ("free will", "existence of god") map to a canonical contested
claim — via the built-in table, then a cached LLM call, then verbatim as a
last resort.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, TOPIC_CLAIMS
from api.stance import _classify_via_cli_text, _query_hash, _sdk_available, _complete_via_sdk

# A query is claim-shaped if it contains a predicate marker; otherwise we
# treat it as a topic.
CLAIM_MARKERS = re.compile(
    r"\b(is|are|was|were|be|being|been|has|have|had|does|do|did|makes?|made|"
    r"exists?|should|ought|must|can|cannot|could|will|would|may|might|"
    r"implies|requires?|proves?|disproves?|means?|matters?|justifies|deserves?|"
    r"depends?|entails?|presupposes?)\b",
    re.I,
)

TOPIC_PROMPT = (
    "Give the single most canonical, contested philosophical claim for the topic "
    '"{topic}" — the thesis philosophers have most centrally argued for and against. '
    "Phrase it as one short declarative sentence. Reply with ONLY the claim, no quotes."
)


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower().rstrip("?.!"))


def resolve_claim(query: str) -> tuple[str, bool, str | None]:
    """Return (claim, was_topic, error). error is set when a topic could not
    be resolved and the query is used verbatim."""
    norm = _normalize(query)
    if norm in TOPIC_CLAIMS:
        return TOPIC_CLAIMS[norm], True, None
    if CLAIM_MARKERS.search(norm) or len(norm.split()) > 6:
        return query, False, None

    # Unknown topic: check the cache, then ask the LLM.
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS topic_claims (query_hash TEXT PRIMARY KEY, claim TEXT NOT NULL)"
    )
    qh = _query_hash(norm)
    row = con.execute("SELECT claim FROM topic_claims WHERE query_hash = ?", (qh,)).fetchone()
    if row:
        con.close()
        return row[0], True, None
    prompt = TOPIC_PROMPT.format(topic=query.strip())
    try:
        text = _complete_via_sdk(prompt) if _sdk_available() else _classify_via_cli_text(prompt)
        claim = text.strip().strip('"').splitlines()[0].strip()
        if not (3 <= len(claim.split()) <= 25):
            raise ValueError(f"implausible claim: {claim!r}")
        con.execute("INSERT OR REPLACE INTO topic_claims VALUES (?, ?)", (qh, claim))
        con.commit()
        con.close()
        return claim, True, None
    except Exception as e:  # noqa: BLE001
        con.close()
        return query, False, f"topic resolution unavailable: {e}"
