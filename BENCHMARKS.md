# Benchmarks — where the index earns its keep

The headline result in the README is uncomfortable on its own: on the real
3,723-passage corpus, **brute force beats the HNSW index**. One vectorised matrix
product over 3,723 rows is hard to beat, and a proximity graph walked in Python
is not going to beat it. That is a real measurement and it invites the obvious
question — *so why build the index at all?*

This is the answer, measured rather than asserted. Brute force costs O(n·d) per
query; a proximity graph costs roughly O(log n). There is a corpus size where the
lines cross, and it is small.

```bash
python -m evals.scale_bench                  # the full sweep
python -m evals.scale_bench --ef-sweep 100000  # the recall/latency knob
```

Every number below came off this machine. Nothing is extrapolated.

---

## The crossover

Median query latency, k=10, 100 queries per configuration.

| vectors | brute force | hnswlib (C++) | PyHNSW (from scratch) |
|--------:|------------:|--------------:|----------------------:|
| 3,723 *(the real corpus)* | **0.187 ms** | 0.180 ms | 0.434 ms |
| 5,000 | 0.260 ms | **0.122 ms** | 0.318 ms |
| 7,500 | 0.496 ms | **0.138 ms** | **0.340 ms** |
| 10,000 | 0.561 ms | 0.136 ms | 0.360 ms |
| 15,000 | 1.135 ms | 0.158 ms | 0.396 ms |
| 20,000 | 1.514 ms | 0.180 ms | 0.411 ms |
| 30,000 | 2.322 ms | 0.193 ms | 0.556 ms |
| 100,000 | 8.442 ms | 0.218 ms | 0.520 ms |
| 300,000 | 27.172 ms | 1.146 ms | — |
| 1,000,000 *(swapping — see below)* | 167.410 ms | 89.450 ms | — |

Brute force grows linearly and the graphs barely move. As a speed-up over exact
search:

| vectors | hnswlib | PyHNSW |
|--------:|--------:|-------:|
| 3,723 | 1.04× | 0.43× |
| 5,000 | 2.13× | 0.82× |
| 7,500 | 3.59× | **1.46×** |
| 30,000 | 12.0× | 4.18× |
| 100,000 | **38.7×** | **16.2×** |
| 300,000 | 23.7× | — |
| 1,000,000 *(memory-bound)* | 1.9× | — |

**The C++ index crosses over at about 4,000 vectors — essentially the size of the
current corpus. The from-scratch Python one crosses over between 5,000 and
7,500.** So the index is not a science project that will pay off one day: the
corpus is already at the point where it starts winning, and one more book tips it.

Past that the gap widens fast. At 100,000 vectors — about 27 books at this
chunking rate — exact search takes 8.4 ms per query and the from-scratch index
takes 0.52 ms.

## Recall

Recall@10 against exact search, at the default `ef_search = 64`:

| vectors | hnswlib | PyHNSW |
|--------:|--------:|-------:|
| 3,723 | 0.993 | 0.999 |
| 10,000 | 0.997 | 1.000 |
| 30,000 | 0.999 | 1.000 |
| 100,000 | 1.000 | 1.000 |
| 300,000 | 0.999 | — |
| 1,000,000 | 0.980 | — |

Approximate search is, on this data, not measurably approximate. That is a
statement about the data as much as the index — real embedding neighbourhoods are
tightly clustered, and a graph walk finds them.

## The build cost, which is where the from-scratch index actually loses

| vectors | hnswlib | PyHNSW | ratio |
|--------:|--------:|-------:|------:|
| 3,723 | 0.3 s | 17.7 s | 59× |
| 30,000 | 3.8 s | 187.3 s | 49× |
| 100,000 | 17.3 s | **855.5 s** (14 min) | 49× |

Roughly fifty times slower to build, consistently, which is about what a
pure-Python inner loop costs against optimised C++. Queries are within 2–3× of
hnswlib because a query touches a few hundred nodes while a build touches
millions.

This is the honest summary of the from-scratch implementation: **the algorithm is
right — recall matches, query latency is the same order — and the constant factor
is Python's.** For a corpus that is rebuilt occasionally and queried constantly,
that trade is survivable up to ~100k vectors and stops being survivable somewhere
past it.

## Where this machine gives out

At 1,000,000 vectors the working set — 1.5 GB of float32 vectors plus the graph —
stops fitting in an 8 GB laptop that is also running Postgres and a browser. The
numbers say so plainly:

| vectors | brute force p50 | hnswlib p50 | speed-up | hnswlib recall |
|--------:|----------------:|------------:|---------:|---------------:|
| 300,000 | 27.2 ms | 1.15 ms | **23.7×** | 0.999 |
| 1,000,000 | 167.4 ms | 89.5 ms | **1.9×** | 0.980 |

Recall stays at 0.980, so the index is still finding the right neighbours — it is
not broken, it is *swapping*. A graph traversal jumps to arbitrary nodes, which is
the access pattern virtual memory handles worst, and 89 ms for a query that took
1.15 ms at 300k is the sound of every hop hitting disk. The build took 31 minutes
for the same reason.

So the 1M row measures this laptop's RAM, not either algorithm. It is recorded
here rather than quietly dropped, but **the honest measured range of this
benchmark is up to 300,000 vectors** — and the real lesson is the one every
vector-search deployment eventually learns: the index has to fit in memory, and
when it stops fitting, the asymptotics stop mattering.

---

## Method, and one mistake worth documenting

- **k=10, 100 queries per configuration**, queries drawn from the corpus itself.
  Latency is the median of those 100; recall is the mean overlap with exact
  search.
- **Every (size, index) pair runs in its own process.** A 1M × 384 float32 matrix
  is 1.5 GB; measuring the next configuration in a process still holding the last
  one measures swap.
- **`M=16`, `ef_construction=200`, `ef_search=64` for both graph indexes**, so the
  comparison is like-for-like rather than a parameter contest.
- **Only the 3,723 row is real data.** Everything above it is synthetic, because
  there is no honest way to conjure a million genuine philosophy passages.

That last point is where the first version of this benchmark went wrong, and the
failure is worth keeping. Synthetic vectors were drawn around cluster centres
with Gaussian noise at `sigma = 0.35`, which sounds tight and is not: in 384
dimensions the noise has 384 directions to spread through while the centre
contributes one, so points landed at cosine **0.14** from their own cluster
centre. That is nearly orthogonal — statistically indistinguishable from uniform
random data, which is the adversarial worst case for any proximity graph.

The benchmark duly reported recall falling to **0.417 at 100k** and **0.204 at
300k**, and an `ef` sweep from 16 to 512 only dragged it from 0.21 to 0.58 —
which looked like a real and alarming finding about HNSW at scale.

It was a finding about the generator. Measuring the real corpus gave the number
that settled it: a real passage sits at cosine **0.839** from its nearest
neighbour, not 0.14. Re-calibrating so the synthetic data has the same
neighbourhood geometry:

| sigma | 0.35 | 0.10 | 0.05 | 0.04 | **0.033** | 0.025 |
|---|---|---|---|---|---|---|
| cosine to own centre | 0.144 | 0.456 | 0.714 | 0.787 | **0.840** | 0.898 |

At `sigma = 0.033` the synthetic corpus reproduces the real corpus's geometry
(nearest neighbour 0.749 vs 0.839, tenth 0.733 vs 0.806), and recall at 100k goes
from 0.417 to **1.000**. The index had been fine the whole time.

The lesson is not subtle and applies to any benchmark on generated data: **if you
do not calibrate the synthetic distribution against the real one, you are
measuring your random number generator.** The check that catches it costs one
line — compare nearest-neighbour similarity in your synthetic data against your
real data — and without it this file would have confidently published a
completely false claim about approximate search.
