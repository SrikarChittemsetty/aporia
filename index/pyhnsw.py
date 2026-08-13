"""A from-scratch HNSW (Hierarchical Navigable Small World) index in NumPy.

This is the Phase 2 depth piece: the same algorithm hnswlib implements in
C++ (Malkov & Yashunin 2016), written from first principles behind the same
`VectorIndex` interface so it can serve as a drop-in replacement.

The structure is a stack of proximity graphs. Every element lands on level 0;
each higher level keeps an exponentially thinning subset (geometric level
assignment with factor 1/ln(M)). A query greedily descends the sparse upper
layers — each hop roughly halves the remaining distance — and runs a beam
search (width `ef`) only on the dense bottom layer. That gives ~O(log n)
hops per query instead of the O(n) scan a flat index needs.

Distances are cosine (vectors are normalized once at build time, so cosine
distance = 1 - dot product). Neighbor selection uses the paper's heuristic
(Algorithm 4): a candidate is kept only if it is closer to the query point
than to any already-selected neighbor, which keeps edges spread across
directions instead of clustering.
"""
from __future__ import annotations

import heapq
import json
import math
import pickle
from pathlib import Path

import numpy as np


class PyHNSW:
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200,
                 ef_search: int = 64, seed: int = 42):
        self.dim = dim
        self.m = m                       # max links per node on levels > 0
        self.m0 = 2 * m                  # max links on level 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._ml = 1.0 / math.log(m)
        self._rng = np.random.default_rng(seed)

        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._levels: list[int] = []             # top level of each node
        self._links: list[list[list[int]]] = []  # _links[node][level] -> neighbor ids
        self._entry: int | None = None           # entry point (highest-level node)

    # -- distance helpers ---------------------------------------------------

    def _dist(self, q: np.ndarray, idx: int) -> float:
        return 1.0 - float(self._vectors[idx] @ q)

    def _dists(self, q: np.ndarray, idxs: list[int]) -> np.ndarray:
        return 1.0 - self._vectors[idxs] @ q

    # -- core search primitives (Algorithms 2 and 4 of the paper) -----------

    def _greedy_descend(self, q: np.ndarray, start: int, level: int) -> int:
        """Single-hop greedy search on one level: move to the closest
        neighbor until no neighbor improves."""
        cur, cur_d = start, self._dist(q, start)
        improved = True
        while improved:
            improved = False
            neighbors = self._links[cur][level]
            if neighbors:
                ds = self._dists(q, neighbors)
                j = int(np.argmin(ds))
                if ds[j] < cur_d:
                    cur, cur_d = neighbors[j], float(ds[j])
                    improved = True
        return cur

    def _search_layer(self, q: np.ndarray, entry: int, ef: int, level: int) -> list[tuple[float, int]]:
        """Beam search with width ef on one level. Returns (dist, id) sorted
        ascending."""
        d0 = self._dist(q, entry)
        visited = {entry}
        candidates = [(d0, entry)]              # min-heap by distance
        best: list[tuple[float, int]] = [(-d0, entry)]  # max-heap (negated)
        while candidates:
            d, node = heapq.heappop(candidates)
            if d > -best[0][0]:
                break                            # nearest candidate is worse than the worst kept
            fresh = [n for n in self._links[node][level] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for nd, n in zip(self._dists(q, fresh), fresh):
                nd = float(nd)
                if len(best) < ef or nd < -best[0][0]:
                    heapq.heappush(candidates, (nd, n))
                    heapq.heappush(best, (-nd, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-negd, n) for negd, n in best)

    def _select_neighbors(self, q: np.ndarray, candidates: list[tuple[float, int]],
                          m: int) -> list[int]:
        """Heuristic selection: keep a candidate only if it's closer to q than
        to every already-kept neighbor (spreads edges across directions)."""
        selected: list[int] = []
        for d, c in candidates:                  # candidates sorted ascending
            if len(selected) >= m:
                break
            if all(d < 1.0 - float(self._vectors[c] @ self._vectors[s]) for s in selected):
                selected.append(c)
        # Backfill with nearest rejects if the heuristic was too strict.
        if len(selected) < m:
            chosen = set(selected)
            for d, c in candidates:
                if len(selected) >= m:
                    break
                if c not in chosen:
                    selected.append(c)
        return selected

    # -- construction -------------------------------------------------------

    def add_items(self, vectors: np.ndarray, ids: list[int] | None = None) -> None:
        """Insert vectors one by one. `ids` must be 0..n-1 order-aligned if
        given (external id mapping is the caller's job — see PyHnswIndex)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        # Preallocate: appending row-by-row would copy the matrix per insert.
        self._vectors = np.vstack([self._vectors, vectors])

        for vec in vectors:
            node = len(self._levels)
            level = int(-math.log(self._rng.random()) * self._ml)
            self._levels.append(level)
            self._links.append([[] for _ in range(level + 1)])

            if self._entry is None:
                self._entry = node
                continue

            # Descend greedily through levels above the new node's level.
            cur = self._entry
            for lv in range(self._levels[self._entry], level, -1):
                if lv <= self._levels[cur]:
                    cur = self._greedy_descend(vec, cur, min(lv, self._levels[cur]))

            # Beam-search and link on each level the new node occupies.
            for lv in range(min(level, self._levels[self._entry]), -1, -1):
                found = self._search_layer(vec, cur, self.ef_construction, lv)
                cap = self.m0 if lv == 0 else self.m
                neighbors = self._select_neighbors(vec, found, cap)
                self._links[node][lv] = list(neighbors)
                for n in neighbors:
                    links = self._links[n][lv]
                    links.append(node)
                    if len(links) > cap:
                        # Re-select the neighbor's links with the same heuristic.
                        nvec = self._vectors[n]
                        ds = 1.0 - self._vectors[links] @ nvec
                        ranked = sorted(zip(ds.tolist(), links))
                        self._links[n][lv] = self._select_neighbors(nvec, ranked, cap)
                cur = found[0][1]

            if level > self._levels[self._entry]:
                self._entry = node

    # -- query --------------------------------------------------------------

    def knn_query(self, q: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        q = np.asarray(q, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        cur = self._entry
        for lv in range(self._levels[self._entry], 0, -1):
            cur = self._greedy_descend(q, cur, lv)
        found = self._search_layer(q, cur, max(self.ef_search, k), 0)[:k]
        return [n for _, n in found], [d for d, _ in found]

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        np.savez_compressed(path.with_suffix(".npz"), vectors=self._vectors,
                            levels=np.asarray(self._levels))
        with open(path.with_suffix(".graph"), "wb") as f:
            pickle.dump({"links": self._links, "entry": self._entry,
                         "m": self.m, "ef_construction": self.ef_construction,
                         "ef_search": self.ef_search}, f)

    @classmethod
    def load(cls, path: Path) -> "PyHNSW":
        data = np.load(path.with_suffix(".npz"))
        with open(path.with_suffix(".graph"), "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=data["vectors"].shape[1], m=meta["m"],
                  ef_construction=meta["ef_construction"], ef_search=meta["ef_search"])
        obj._vectors = data["vectors"]
        obj._levels = data["levels"].tolist()
        obj._links = meta["links"]
        obj._entry = meta["entry"]
        return obj
