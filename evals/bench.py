"""Benchmark the three vector indexes on the real corpus embeddings.

Usage: python -m evals.bench
Requires data/index/vectors.npz (written by index.build_index).

Protocol: hold out 100 corpus vectors as queries, build each index on the
remainder, measure recall@10 against exact brute-force search, plus build
time and per-query latency. Prints a Markdown table for the README.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import INDEX_DIR
from index.vector_index import HnswlibIndex, NumpyIndex, PyHnswIndex

K = 10
N_QUERIES = 100


def main() -> None:
    data = np.load(INDEX_DIR / "vectors.npz")
    vectors, ids = data["vectors"], data["ids"].tolist()

    rng = np.random.default_rng(7)
    q_idx = rng.choice(len(ids), size=N_QUERIES, replace=False)
    mask = np.ones(len(ids), dtype=bool)
    mask[q_idx] = False
    build_vecs = vectors[mask]
    build_ids = [i for i, keep in zip(ids, mask) if keep]
    queries = vectors[q_idx]

    candidates = [
        ("NumPy brute force (exact)", NumpyIndex()),
        ("hnswlib (C++)", HnswlibIndex()),
        ("PyHNSW (ours, NumPy)", PyHnswIndex()),
    ]

    results = []
    truth: list[set] = []
    for name, index in candidates:
        t0 = time.perf_counter()
        index.build(build_vecs, build_ids)
        build_s = time.perf_counter() - t0

        latencies, recalls = [], []
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            got, _ = index.search(q, K)
            latencies.append((time.perf_counter() - t0) * 1000)
            if name.startswith("NumPy"):
                truth.append(set(got))
            else:
                recalls.append(len(truth[qi] & set(got)) / K)
        lat = np.asarray(latencies)
        recall = 1.0 if name.startswith("NumPy") else float(np.mean(recalls))
        results.append((name, recall, build_s, float(np.percentile(lat, 50)),
                        float(np.percentile(lat, 95))))

    n = len(build_ids)
    print(f"\ncorpus: {n} vectors, dim {vectors.shape[1]}, {N_QUERIES} held-out queries, k={K}\n")
    print("| Index | recall@10 | build time | p50 query | p95 query |")
    print("|---|---|---|---|---|")
    for name, recall, build_s, p50, p95 in results:
        print(f"| {name} | {recall:.3f} | {build_s:.1f}s | {p50:.2f} ms | {p95:.2f} ms |")


if __name__ == "__main__":
    main()
