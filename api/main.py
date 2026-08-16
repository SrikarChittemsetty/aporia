"""Aporia API: search primary-source philosophy by argument.

Run: uvicorn api.main:app --reload --port 8080  (from the project root)
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, EMBED_MODEL, INDEX_DIR, QUERY_PREFIX, ROOT, TOP_K
from api import expand, stance
from api.claims import resolve_claim
from index.vector_index import load_index

app = FastAPI(title="Aporia", description="Search philosophy by argument, not keyword.")

_state: dict = {}


@app.on_event("startup")
def startup() -> None:
    from sentence_transformers import SentenceTransformer

    _state["model"] = SentenceTransformer(EMBED_MODEL)
    _state["index"] = load_index(INDEX_DIR)


def _fetch_chunks(ids: list[int]) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    rows = con.execute(
        f"SELECT id, author, work, citation_path, text FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    con.close()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


@app.get("/search")
def search(q: str = Query(..., min_length=3), k: int = TOP_K):
    # Bare topics ("free will") resolve to a canonical claim; retrieval still
    # uses the original query for breadth, stance is judged against the claim.
    claim, was_topic, claim_error = resolve_claim(q)

    # Retrieval searches with a blend of the query and a hypothetical passage
    # written in the corpus's own idiom — see api/expand.py for why, and
    # evals/expansion_eval.py for what it is worth. Degrades to the plain query
    # vector if no LLM backend is reachable.
    vector, _hypothetical, expand_error = expand.search_vector(_state["model"], q)
    ids, scores = _state["index"].search(np.asarray(vector, dtype=np.float32), k=k)
    passages = _fetch_chunks(ids)
    score_by_id = dict(zip(ids, scores))

    stances, stance_error = stance.classify(claim, passages)
    stance_error = stance_error or claim_error or expand_error

    grouped = {"for": [], "against": [], "nuance": []}
    for p in passages:
        s = stances.get(p["id"], {"stance": "nuance", "move": "", "confidence": 0.0})
        grouped[s["stance"]].append({
            "id": p["id"],
            "author": p["author"],
            "work": p["work"],
            "citation": p["citation_path"],
            "text": p["text"],
            "move": s["move"],
            "confidence": s["confidence"],
            "similarity": round(score_by_id.get(p["id"], 0.0), 4),
        })
    return {"query": q, "claim": claim, "was_topic": was_topic,
            "stance_error": stance_error, **grouped}


@app.get("/passage/{chunk_id}")
def passage(chunk_id: int, context: int = 2):
    """Return a chunk plus `context` neighboring chunks from the same work."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if not row:
        con.close()
        raise HTTPException(404, "no such passage")
    neighbors = con.execute(
        "SELECT id, citation_path, seq, text FROM chunks"
        " WHERE gutenberg_id = ? AND seq BETWEEN ? AND ? ORDER BY seq",
        (row["gutenberg_id"], row["seq"] - context, row["seq"] + context),
    ).fetchall()
    con.close()
    return {
        "id": row["id"],
        "author": row["author"],
        "work": row["work"],
        "citation": row["citation_path"],
        "context": [dict(n) for n in neighbors],
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")
