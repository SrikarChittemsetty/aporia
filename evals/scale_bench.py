"""Where does the HNSW index actually start beating brute force?

The existing benchmark answers "is the from-scratch index correct?" on the real
3,723-vector corpus, and there the honest answer is that brute force wins: one
vectorised matrix product over 3,723 rows is faster than walking a proximity
graph in Python. That is a real result and it is also a small-n artefact, and
leaving it there invites the obvious question — *so why build the index at all?*

This measures the answer. Brute force costs O(n·d) per query and the graph costs
roughly O(log n), so there is a corpus size where the lines cross. This finds it
by measuring, not by asserting, across corpus sizes from the real 3.7k up to 1M.

    python -m evals.scale_bench                     # the full sweep
    python -m evals.scale_bench --sizes 3723,10000  # a quick subset

Each (size, index) pair runs in a **separate process**, because a 1M × 384
float32 matrix is 1.5 GB and this machine has 8 GB: measuring the next
configuration in a process that still holds the last one would measure swap.
Results accumulate in evals/scale_results.json so an interrupted sweep resumes.

## On the data

Only the 3,723-vector row uses the real corpus. Everything above it is synthetic,
because there is no honest way to conjure 1M genuine philosophy passages — and
synthetic vectors have to be generated carefully or the benchmark lies. Uniform
random unit vectors are close to orthogonal in 384 dimensions, which is the
adversarial case for any proximity graph and would understate recall badly. Real
embeddings are clustered, so these are too: points are drawn around a set of
random centres with controlled spread, then normalised.

The recall numbers below therefore describe *this distribution*, and the real
corpus row is the anchor that says whether the synthetic one is in the right
neighbourhood. Latency and build time are far less distribution-sensitive; those
transfer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import INDEX_DIR

RESULTS = Path(__file__).parent / "scale_results.json"
DIM = 384
K = 10
N_QUERIES = 100
SEED = 20260815

# The real corpus size comes first so the anchor is measured before anything
# synthetic; sizes past that are powers-ish of ten up to a million.
DEFAULT_SIZES = [3723, 10_000, 30_000, 100_000, 300_000, 1_000_000]

# Build time is the constraint on the pure-Python index, not correctness. Past
# roughly this many vectors its build runs into hours, which is itself one of
# the findings rather than a reason to skip the measurement quietly.
PYHNSW_MAX_N = 100_000

# How tightly synthetic points cluster around their centre, and the single most
# important number in this file.
#
# The first version used 0.35, which felt like a reasonable spread and was not:
# in 384 dimensions a point ends up at cosine 0.14 from its own cluster centre,
# because the noise has 384 dimensions to spread through and the centre only
# contributes one direction. That is nearly orthogonal — indistinguishable from
# uniform random data, which is the adversarial worst case for any proximity
# graph. It made recall look like an index problem when it was a data problem.
#
# Calibrated instead against the real corpus, where a passage sits at cosine
# 0.839 from its nearest neighbour. sigma = 0.033 reproduces that:
#
#     sigma   0.35    0.10    0.05    0.04    0.033   0.025
#     cos     0.144   0.456   0.714   0.787   0.840   0.898
#
# So the synthetic corpus now has the neighbourhood geometry of the real one,
# and recall measured on it means something.
CLUSTER_SIGMA = 0.033


def make_vectors(n: int, dim: int = DIM, seed: int = SEED) -> np.ndarray:
    """Clustered unit vectors — a stand-in for real embedding geometry.

    Deterministic in (n, seed) so every index type sees identical data without
    anyone having to store 1.5 GB of it.
    """
    if n == 3723:
        # The anchor: real corpus embeddings, not a simulation of them.
        data = np.load(INDEX_DIR / "vectors.npz")
        vecs = np.asarray(data["vectors"], dtype=np.float32)
        return vecs[:n]

    rng = np.random.default_rng(seed)
    n_centres = max(16, n // 250)
    centres = rng.normal(size=(n_centres, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    out = np.empty((n, dim), dtype=np.float32)
    step = 50_000
    for start in range(0, n, step):
        end = min(start + step, n)
        which = rng.integers(0, n_centres, size=end - start)
        block = centres[which] + rng.normal(
            scale=CLUSTER_SIGMA, size=(end - start, dim)
        ).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        out[start:end] = block
    return out


def exact_topk(vectors: np.ndarray, queries: np.ndarray, k: int) -> list[set[int]]:
    """Ground truth, computed in chunks so the score matrix never blows up."""
    truth = []
    for q in queries:
        best_scores = np.full(k, -np.inf, dtype=np.float32)
        best_idx = np.zeros(k, dtype=np.int64)
        step = 200_000
        for start in range(0, len(vectors), step):
            block = vectors[start:start + step]
            scores = block @ q
            take = min(k, len(scores))
            part = np.argpartition(-scores, take - 1)[:take]
            cand_scores = np.concatenate([best_scores, scores[part]])
            cand_idx = np.concatenate([best_idx, part + start])
            order = np.argsort(-cand_scores)[:k]
            best_scores, best_idx = cand_scores[order], cand_idx[order]
        truth.append(set(best_idx.tolist()))
    return truth


# --- one (size, index) measurement, run in its own process --------------------


def measure(kind: str, n: int) -> dict:
    from index.vector_index import HnswlibIndex, NumpyIndex, PyHnswIndex

    vectors = make_vectors(n)
    ids = list(range(len(vectors)))

    rng = np.random.default_rng(SEED + 1)
    q_idx = rng.choice(len(vectors), size=min(N_QUERIES, len(vectors)), replace=False)
    queries = vectors[q_idx].copy()

    index = {"numpy": NumpyIndex, "hnswlib": HnswlibIndex, "pyhnsw": PyHnswIndex}[kind]()

    t0 = time.perf_counter()
    index.build(vectors, ids)
    build_s = time.perf_counter() - t0

    # Ground truth for recall. For the exact index this is trivially itself, so
    # skip the work and record recall 1.0 by definition.
    truth = None if kind == "numpy" else exact_topk(vectors, queries, K)

    latencies, recalls = [], []
    for i, q in enumerate(queries):
        t = time.perf_counter()
        got_ids, _ = index.search(q, K)
        latencies.append((time.perf_counter() - t) * 1000)
        if truth is not None:
            recalls.append(len(set(got_ids) & truth[i]) / K)

    latencies.sort()
    return {
        "index": kind,
        "n": n,
        "build_s": round(build_s, 2),
        "p50_ms": round(latencies[len(latencies) // 2], 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
        "recall": round(float(np.mean(recalls)), 4) if recalls else 1.0,
    }


def measure_ef(n: int, efs: list[int]) -> list[dict]:
    """Recall and latency across ef_search, at one corpus size.

    Recall falls as the corpus grows at a fixed `ef`, and the honest question is
    whether that is a property of the index or of one parameter left at its
    default. `ef` is the search-beam width: the knob that trades latency for
    recall. Measuring the curve answers the question instead of arguing it.
    """
    import hnswlib

    vectors = make_vectors(n)
    rng = np.random.default_rng(SEED + 1)
    q_idx = rng.choice(len(vectors), size=N_QUERIES, replace=False)
    queries = vectors[q_idx].copy()
    truth = exact_topk(vectors, queries, K)

    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    index.add_items(vectors, list(range(len(vectors))))

    rows = []
    for ef in efs:
        index.set_ef(ef)
        latencies, recalls = [], []
        for i, q in enumerate(queries):
            t = time.perf_counter()
            labels, _ = index.knn_query(q, k=K)
            latencies.append((time.perf_counter() - t) * 1000)
            recalls.append(len(set(labels[0].tolist()) & truth[i]) / K)
        latencies.sort()
        rows.append({
            "n": n, "ef": ef,
            "p50_ms": round(latencies[len(latencies) // 2], 3),
            "recall": round(float(np.mean(recalls)), 4),
        })
    return rows


# --- orchestration ------------------------------------------------------------


def load_results() -> list[dict]:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return []


def save_results(rows: list[dict]) -> None:
    RESULTS.write_text(json.dumps(rows, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--indexes", default="numpy,hnswlib,pyhnsw")
    ap.add_argument("--pyhnsw-max", type=int, default=PYHNSW_MAX_N)
    ap.add_argument("--child", default=None, help="internal: run one measurement")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-measure existing rows")
    ap.add_argument("--ef-sweep", type=int, default=None,
                    help="instead of the size sweep, measure recall vs ef at this size")
    args = ap.parse_args()

    if args.ef_sweep:
        rows = measure_ef(args.ef_sweep, [16, 32, 64, 128, 256, 512])
        print(f"\nhnswlib at n={args.ef_sweep:,} — the recall/latency knob\n")
        print(f"{'ef':>6} {'p50':>10} {'recall@10':>11}")
        print("-" * 30)
        for r in rows:
            print(f"{r['ef']:>6} {r['p50_ms']:>9.3f}m {r['recall']:>11.3f}")
        out = Path(__file__).parent / "ef_sweep.json"
        out.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"\nwrote {out}")
        return

    if args.child:
        print(json.dumps(measure(args.child, args.n)))
        return

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES
    kinds = args.indexes.split(",")

    rows = [] if args.force else load_results()
    have = {(r["index"], r["n"]) for r in rows}

    for n in sizes:
        for kind in kinds:
            if kind == "pyhnsw" and n > args.pyhnsw_max:
                print(f"  skip  pyhnsw @ {n:>9,} — build time past the budget "
                      f"(--pyhnsw-max {args.pyhnsw_max})")
                continue
            if (kind, n) in have:
                print(f"  have  {kind:<8} @ {n:>9,}")
                continue

            print(f"  run   {kind:<8} @ {n:>9,} …", end="", flush=True)
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, __file__, "--child", kind, "--n", str(n)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
                print(f" FAILED after {time.perf_counter() - t0:.0f}s: {tail[0][:120]}")
                continue

            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            save_results(rows)
            print(f" build {row['build_s']:>8.1f}s  p50 {row['p50_ms']:>7.3f}ms  "
                  f"recall {row['recall']:.3f}")

    # --- the table ------------------------------------------------------------
    print(f"\n{'n':>10} {'index':<10} {'build':>10} {'p50':>10} {'p95':>10} {'recall':>8}")
    print("-" * 62)
    for n in sorted({r["n"] for r in rows}):
        for r in sorted([x for x in rows if x["n"] == n], key=lambda x: x["index"]):
            print(f"{r['n']:>10,} {r['index']:<10} {r['build_s']:>9.1f}s "
                  f"{r['p50_ms']:>9.3f}m {r['p95_ms']:>9.3f}m {r['recall']:>8.3f}")

    # --- where the lines cross ------------------------------------------------
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], {})[r["index"]] = r

    print("\ncrossover (brute force p50 ÷ index p50 — above 1.0 the index wins):")
    for n in sorted(by_n):
        exact = by_n[n].get("numpy")
        if not exact:
            continue
        bits = []
        for kind in ("hnswlib", "pyhnsw"):
            r = by_n[n].get(kind)
            if r and r["p50_ms"]:
                bits.append(f"{kind} {exact['p50_ms'] / r['p50_ms']:.2f}x")
        if bits:
            print(f"  {n:>10,}  " + "   ".join(bits))

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
