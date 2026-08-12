"""Vector index behind a swappable interface.

Phase 0 ships HnswlibIndex (library). Phase 2 replaces it with our own HNSW
implementation behind the same interface — keep this contract stable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorIndex:
    """Interface: build(vectors, ids), search(vector, k) -> (ids, scores)."""

    def build(self, vectors: np.ndarray, ids: list[int]) -> None:
        raise NotImplementedError

    def search(self, vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        raise NotImplementedError

    def save(self, dirpath: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, dirpath: Path) -> "VectorIndex":
        raise NotImplementedError


class HnswlibIndex(VectorIndex):
    def __init__(self, dim: int | None = None):
        self.dim = dim
        self._index = None

    def build(self, vectors: np.ndarray, ids: list[int]) -> None:
        import hnswlib

        self.dim = vectors.shape[1]
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.init_index(max_elements=len(ids), ef_construction=200, M=16)
        self._index.add_items(vectors, ids)
        self._index.set_ef(64)

    def search(self, vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        labels, distances = self._index.knn_query(vector, k=k)
        # cosine distance -> similarity
        return labels[0].tolist(), (1.0 - distances[0]).tolist()

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(dirpath / "hnsw.bin"))
        (dirpath / "meta.json").write_text(json.dumps({"dim": self.dim, "kind": "hnswlib"}))

    @classmethod
    def load(cls, dirpath: Path) -> "HnswlibIndex":
        import hnswlib

        meta = json.loads((dirpath / "meta.json").read_text())
        obj = cls(dim=meta["dim"])
        obj._index = hnswlib.Index(space="cosine", dim=meta["dim"])
        obj._index.load_index(str(dirpath / "hnsw.bin"))
        obj._index.set_ef(64)
        return obj


class NumpyIndex(VectorIndex):
    """Brute-force cosine fallback (also the recall baseline for Phase 2 benchmarks)."""

    def __init__(self):
        self._vectors = None
        self._ids = None

    def build(self, vectors: np.ndarray, ids: list[int]) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.clip(norms, 1e-12, None)
        self._ids = np.asarray(ids)

    def search(self, vector: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        v = vector / max(np.linalg.norm(vector), 1e-12)
        sims = self._vectors @ v
        top = np.argsort(-sims)[:k]
        return self._ids[top].tolist(), sims[top].tolist()

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        np.savez(dirpath / "numpy_index.npz", vectors=self._vectors, ids=self._ids)
        (dirpath / "meta.json").write_text(json.dumps({"kind": "numpy"}))

    @classmethod
    def load(cls, dirpath: Path) -> "NumpyIndex":
        data = np.load(dirpath / "numpy_index.npz")
        obj = cls()
        obj._vectors = data["vectors"]
        obj._ids = data["ids"]
        return obj


def load_index(dirpath: Path) -> VectorIndex:
    meta = json.loads((dirpath / "meta.json").read_text())
    if meta.get("kind") == "hnswlib":
        return HnswlibIndex.load(dirpath)
    return NumpyIndex.load(dirpath)
