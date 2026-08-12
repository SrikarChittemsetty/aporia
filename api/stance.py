"""Stance + argument-move classification for retrieved passages.

One batched call per query classifies every passage as for/against/nuance
relative to the claim, with a one-line description of the argumentative move.
Results are cached in SQLite keyed by (query, chunk).

Backends:
  - Anthropic SDK (used when ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN or an
    `ant auth login` profile is available)
  - local `claude` CLI subprocess (fallback for dev machines authenticated
    through Claude Code)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ANTHROPIC_MODEL, DB_PATH

VALID_STANCES = {"for", "against", "nuance"}

PROMPT_TEMPLATE = """You are a philosophy research assistant classifying primary-source passages by their dialectical stance toward a claim.

Claim under examination: "{query}"

For each passage below, decide:
- stance: "for" if the passage argues in favor of the claim, "against" if it argues against it, "nuance" if it reframes, qualifies, or dissolves the question — or if you are unsure.
- move: one short sentence naming the argumentative move the passage makes (e.g. "Defines liberty as acting according to one's will, making free will compatible with necessity"). Ground it in the passage itself; never invent content.
- confidence: 0.0-1.0.

Passages:
{passages}

Respond with ONLY a JSON array, one object per passage, like:
[{{"id": 1, "stance": "for", "move": "...", "confidence": 0.8}}, ...]
"""


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _render_passages(passages: list[dict]) -> str:
    parts = []
    for p in passages:
        text = p["text"][:1500]
        parts.append(f'[{p["id"]}] {p["author"]}, {p["work"]} ({p["citation_path"]}):\n"{text}"')
    return "\n\n".join(parts)


def _extract_json_array(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON array in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def _sdk_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    cfg = Path.home() / ".config" / "anthropic"
    return cfg.exists() and any(cfg.glob("credentials/*.json"))


def _classify_via_sdk(prompt: str) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model refused the classification request")
    text = next(b.text for b in response.content if b.type == "text")
    return _extract_json_array(text)


def _classify_via_cli(prompt: str) -> list[dict]:
    claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    result = subprocess.run(
        [claude, "-p", prompt],
        capture_output=True, text=True, timeout=300,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[:300]
        raise RuntimeError(f"claude CLI failed: {detail}")
    return _extract_json_array(result.stdout)


def classify(query: str, passages: list[dict]) -> tuple[dict[int, dict], str | None]:
    """Return ({chunk_id: {stance, move, confidence}}, error).

    Uses the cache where possible; classifies the rest in one batched call.
    If no LLM backend is reachable, returns uncached "nuance" fallbacks plus
    an error string — failures are never written to the cache.
    """
    qh = _query_hash(query)
    con = sqlite3.connect(DB_PATH)
    results: dict[int, dict] = {}
    for cid, stance, move, conf in con.execute(
        "SELECT chunk_id, stance, move, confidence FROM stance_cache WHERE query_hash = ?", (qh,)
    ):
        results[cid] = {"stance": stance, "move": move, "confidence": conf}

    error: str | None = None
    missing = [p for p in passages if p["id"] not in results]
    if missing:
        prompt = PROMPT_TEMPLATE.format(query=query, passages=_render_passages(missing))
        try:
            if _sdk_available():
                raw = _classify_via_sdk(prompt)
            else:
                raw = _classify_via_cli(prompt)
        except Exception as e:  # noqa: BLE001
            error = f"stance backend unavailable: {e}"
            for p in missing:
                results[p["id"]] = {"stance": "nuance", "move": "", "confidence": 0.0}
            con.close()
            return results, error
        by_id = {int(r["id"]): r for r in raw if "id" in r}
        for p in missing:
            r = by_id.get(p["id"])
            stance = (r or {}).get("stance", "nuance")
            if stance not in VALID_STANCES:
                stance = "nuance"
            entry = {
                "stance": stance,
                "move": (r or {}).get("move", ""),
                "confidence": float((r or {}).get("confidence", 0.0)),
            }
            results[p["id"]] = entry
            con.execute(
                "INSERT OR REPLACE INTO stance_cache (query_hash, chunk_id, stance, move, confidence)"
                " VALUES (?,?,?,?,?)",
                (qh, p["id"], entry["stance"], entry["move"], entry["confidence"]),
            )
        con.commit()
    con.close()
    return results, error
