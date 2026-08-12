"""Embed all chunks and build the vector index.

Usage: python -m index.build_index
Reads chunks from SQLite, writes the index to data/index/.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, EMBED_MODEL, INDEX_DIR
from index.vector_index import HnswlibIndex, NumpyIndex


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    con.close()
    if not rows:
        raise SystemExit("no chunks in DB — run ingest.chunk first")
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"embedding {len(texts)} chunks with {EMBED_MODEL} ...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    try:
        index = HnswlibIndex()
        index.build(vectors, ids)
        kind = "hnswlib"
    except ImportError:
        index = NumpyIndex()
        index.build(vectors, ids)
        kind = "numpy (hnswlib unavailable)"
    index.save(INDEX_DIR)
    print(f"built {kind} index over {len(ids)} vectors -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
