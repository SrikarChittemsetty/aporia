"""Embed all chunks and build the vector index.

Usage: python -m index.build_index [--reembed]
Reads chunks from SQLite, writes the index (kind = config.INDEX_KIND) to
data/index/. Embeddings are cached in data/index/vectors.npz and reused when
they still match the chunk table, so switching index kinds takes seconds;
pass --reembed to force re-encoding.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, EMBED_MODEL, INDEX_DIR, INDEX_KIND
from index.vector_index import HnswlibIndex, NumpyIndex, PyHnswIndex

KINDS = {"hnswlib": HnswlibIndex, "pyhnsw": PyHnswIndex, "numpy": NumpyIndex}


def _cached_vectors(ids: list[int]) -> np.ndarray | None:
    path = INDEX_DIR / "vectors.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if data["ids"].tolist() == ids:
        return data["vectors"]
    return None


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    con.close()
    if not rows:
        raise SystemExit("no chunks in DB — run ingest.chunk first")
    ids = [r[0] for r in rows]

    vectors = None if "--reembed" in sys.argv else _cached_vectors(ids)
    if vectors is None:
        texts = [r[1] for r in rows]
        print(f"embedding {len(texts)} chunks with {EMBED_MODEL} ...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
        vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(INDEX_DIR / "vectors.npz", vectors=vectors,
                            ids=np.asarray(ids))
    else:
        print(f"reusing cached embeddings for {len(ids)} chunks")

    index = KINDS[INDEX_KIND]()
    index.build(vectors, ids)
    index.save(INDEX_DIR)
    print(f"built {INDEX_KIND} index over {len(ids)} vectors -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
