"""Bounds on what one caller can make this service do.

`/search` is not a cheap endpoint. A single request embeds a query, walks the
index, and — for any claim not already cached — sends every retrieved passage to
Claude in one batched call. That means an unauthenticated stranger can spend the
operator's money, and the amount they can spend per request was, until this
module existed, unbounded: `k` came straight off the query string.

Three limits, each closing a different hole:

  * `clamp_k` — a caller asked for 10,000 passages and got them. The LLM prompt
    is built from the retrieved passages, so `k` is a direct multiplier on token
    spend. It is now clamped to MAX_K.
  * query length — a very long "claim" is a cheap way to inflate the prompt,
    since the claim is interpolated into it once per classification call.
  * a per-client rate limit — the one that matters most, because caching only
    helps for *repeated* claims. A script sending distinct nonsense claims
    misses the cache every time and bills a fresh call each request.

The limiter is in-process and per-instance on purpose. This deploys as a single
instance; a shared counter would mean Redis, which is a dependency and a cost to
solve a problem this deployment does not have. If it ever runs behind more than
one instance, the limit becomes per-instance and this comment becomes a bug
report.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

# Retrieval breadth. TOP_K is 12; allowing a little headroom for experimentation
# without letting anyone build a 10,000-passage prompt.
MAX_K = 50
MIN_K = 1

# A claim is a sentence. Anything past this is not a claim, it is a payload.
MAX_QUERY_CHARS = 300

# Per-client budget. Generous for a human clicking around a demo, hopeless for a
# script trying to drain an API balance.
WINDOW_SECONDS = 60.0
MAX_REQUESTS_PER_WINDOW = 20


def clamp_k(k: int) -> int:
    """Keep retrieval breadth inside sane bounds. `k` multiplies LLM spend."""
    return max(MIN_K, min(int(k), MAX_K))


class SlidingWindowLimiter:
    """Fixed budget of requests per rolling window, per key.

    A sliding window rather than a fixed one: a fixed window lets a caller send
    the full budget at 0:59 and again at 1:01, which is double the intended rate
    at exactly the moment it matters. Keeping timestamps costs a deque per
    client and answers the question exactly.
    """

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS_PER_WINDOW,
        window_seconds: float = WINDOW_SECONDS,
        sweep_threshold: int = 1024,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._sweep_threshold = sweep_threshold

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.max_requests:
                # Retry when the oldest hit falls out of the window.
                return False, max(0.0, hits[0] + self.window_seconds - now)

            hits.append(now)

            # Clients that stopped calling must not accumulate for ever. Only
            # the key being checked gets its timestamps pruned above, so without
            # this the map grows once per distinct client and never shrinks —
            # and since the key can come from a spoofable header, that is an
            # unbounded-memory hole rather than untidiness. Sweeping every key
            # is O(clients), so it only runs when the map is actually large.
            if len(self._hits) > self._sweep_threshold:
                self._sweep(cutoff)

            return True, 0.0

    def _sweep(self, cutoff: float) -> None:
        """Drop expired timestamps everywhere, then forget idle clients.

        Caller holds the lock.
        """
        for key in list(self._hits):
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if not hits:
                del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_key(request) -> str:
    """Identify the caller.

    Behind a load balancer the socket address is the balancer, so the real
    client is the first entry of X-Forwarded-For. Trusting that header is only
    safe *because* this is deployed behind a balancer that overwrites it —
    exposed directly to the internet it would be trivially spoofable, and the
    limit would be decorative.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
