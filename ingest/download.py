"""Download the Phase 0 corpus from Project Gutenberg and strip boilerplate.

Usage: python -m ingest.download
Writes cleaned plain text to data/raw/<gutenberg_id>.txt
"""
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CORPUS, RAW_DIR

MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

START_RE = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", re.I)
END_RE = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", re.I)


def fetch(gid: int) -> str:
    last_err = None
    for tmpl in MIRRORS:
        url = tmpl.format(id=gid)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aporia-ingest/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8-sig", errors="replace")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"could not fetch Gutenberg #{gid}: {last_err}")


def strip_boilerplate(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for entry in CORPUS:
        gid = entry["gutenberg_id"]
        out = RAW_DIR / f"{gid}.txt"
        if out.exists():
            print(f"#{gid} already downloaded ({out})")
            continue
        print(f"downloading #{gid}: {entry['author']} — {entry['work']}")
        text = strip_boilerplate(fetch(gid))
        out.write_text(text, encoding="utf-8")
        print(f"  saved {len(text):,} chars -> {out}")


if __name__ == "__main__":
    main()
