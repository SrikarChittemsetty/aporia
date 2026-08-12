"""Chunk raw texts into ~100-300 word argument units with citation metadata.

Usage: python -m ingest.chunk
Reads data/raw/<id>.txt, writes chunks into SQLite at data/aporia.db.
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS, CORPUS, DB_PATH, RAW_DIR

# Heading heuristics for Gutenberg plain text: SECTION/CHAPTER/PART/BOOK lines,
# roman-numeral headings, or short ALL-CAPS lines.
HEADING_RE = re.compile(
    r"^(?:(?:SECTION|CHAPTER|PART|BOOK|ESSAY|PREFACE|INTRODUCTION|APPENDIX|PROP\.?|PROPOSITION)\b.*"
    r"|[IVXLC]+\.?(?:\s+.*)?"
    r"|[A-Z][A-Z0-9 ,.'\-:;]{3,60})$"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    gutenberg_id INTEGER NOT NULL,
    author TEXT NOT NULL,
    work TEXT NOT NULL,
    citation_path TEXT NOT NULL,
    seq INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stance_cache (
    query_hash TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    stance TEXT NOT NULL,
    move TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (query_hash, chunk_id)
);
"""


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 70:
        return False
    if line.endswith((".", "!", "?")) and len(line.split()) > 6:
        return False
    return bool(HEADING_RE.match(line)) and not line.islower()


def paragraphs_with_offsets(text: str):
    """Yield (start, end, paragraph_text) over blank-line-separated blocks."""
    pos = 0
    for block in re.split(r"\n\s*\n", text):
        start = text.find(block, pos)
        if start == -1:
            start = pos
        end = start + len(block)
        pos = end
        cleaned = re.sub(r"\s+", " ", block).strip()
        if cleaned:
            yield start, end, cleaned


def chunk_work(text: str):
    """Yield dicts of merged paragraphs forming 100-300 word chunks,
    tracking the most recent heading as the citation path."""
    heading = "front matter"
    buf, buf_start, buf_end = [], None, None

    def flush():
        nonlocal buf, buf_start, buf_end
        if buf:
            yield_text = " ".join(buf)
            out = {
                "citation": heading,
                "char_start": buf_start,
                "char_end": buf_end,
                "text": yield_text,
            }
            buf, buf_start, buf_end = [], None, None
            return out
        return None

    for start, end, para in paragraphs_with_offsets(text):
        if is_heading(para):
            c = flush()
            if c:
                yield c
            heading = para[:70]
            continue
        words_in_buf = sum(len(p.split()) for p in buf)
        para_words = len(para.split())
        if words_in_buf and words_in_buf + para_words > CHUNK_MAX_WORDS:
            c = flush()
            if c:
                yield c
        # Very long single paragraphs get split on sentence boundaries.
        if para_words > CHUNK_MAX_WORDS:
            sentences = re.split(r"(?<=[.!?]) +", para)
            piece, piece_words = [], 0
            for s in sentences:
                piece.append(s)
                piece_words += len(s.split())
                if piece_words >= CHUNK_MIN_WORDS + (CHUNK_MAX_WORDS - CHUNK_MIN_WORDS) // 2:
                    yield {
                        "citation": heading,
                        "char_start": start,
                        "char_end": end,
                        "text": " ".join(piece),
                    }
                    piece, piece_words = [], 0
            if piece:
                yield {
                    "citation": heading,
                    "char_start": start,
                    "char_end": end,
                    "text": " ".join(piece),
                }
            continue
        if not buf:
            buf_start = start
        buf.append(para)
        buf_end = end
    c = flush()
    if c:
        yield c


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    if rebuild:
        # Full rebuild changes chunk ids, so cached stances are invalid too.
        con.execute("DELETE FROM chunks")
        con.execute("DELETE FROM stance_cache")

    # Incremental by default: only chunk works not already in the DB, so
    # existing chunk ids (and the stance cache keyed on them) stay stable.
    existing = {row[0] for row in con.execute("SELECT DISTINCT gutenberg_id FROM chunks")}

    total = 0
    for entry in CORPUS:
        gid = entry["gutenberg_id"]
        if gid in existing:
            print(f"#{gid} {entry['author']}: already chunked, skipping")
            continue
        raw = (RAW_DIR / f"{gid}.txt").read_text(encoding="utf-8")
        seq = 0
        for c in chunk_work(raw):
            # Skip tiny fragments (tables of contents, stray lines).
            if len(c["text"].split()) < 40:
                continue
            con.execute(
                "INSERT INTO chunks (gutenberg_id, author, work, citation_path, seq, char_start, char_end, text)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (gid, entry["author"], entry["work"], c["citation"], seq,
                 c["char_start"], c["char_end"], c["text"]),
            )
            seq += 1
        total += seq
        print(f"#{gid} {entry['author']}: {seq} chunks")
    con.commit()
    con.close()
    print(f"total: {total} chunks -> {DB_PATH}")


if __name__ == "__main__":
    main()
