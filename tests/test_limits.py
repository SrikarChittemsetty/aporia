"""The bounds that stop one caller spending the operator's whole API balance."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import limits


def test_clamp_k_closes_the_expensive_hole():
    """`k` is a direct multiplier on how many passages go into the LLM prompt.
    Before clamping, `?k=10000` was a legal request."""
    assert limits.clamp_k(10_000) == limits.MAX_K
    assert limits.clamp_k(0) == limits.MIN_K
    assert limits.clamp_k(-5) == limits.MIN_K
    # Ordinary values pass through untouched.
    assert limits.clamp_k(12) == 12


def test_limiter_allows_up_to_the_budget_then_refuses():
    lim = limits.SlidingWindowLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = lim.check("1.2.3.4", now=100.0)
        assert allowed
    allowed, retry_after = lim.check("1.2.3.4", now=100.0)
    assert not allowed
    assert retry_after == pytest.approx(60.0)


def test_clients_are_budgeted_separately():
    lim = limits.SlidingWindowLimiter(max_requests=1, window_seconds=60)
    assert lim.check("1.1.1.1", now=0.0)[0]
    assert not lim.check("1.1.1.1", now=0.0)[0]
    # A different caller is unaffected by the first one's spending.
    assert lim.check("2.2.2.2", now=0.0)[0]


def test_the_window_actually_slides():
    """A fixed window would let a caller send the full budget at 0:59 and again
    at 1:01 — double the intended rate exactly when it matters."""
    lim = limits.SlidingWindowLimiter(max_requests=2, window_seconds=10)
    assert lim.check("c", now=0.0)[0]
    assert lim.check("c", now=5.0)[0]
    assert not lim.check("c", now=9.0)[0]

    # At t=11 the first hit has aged out, so exactly one slot frees up.
    assert lim.check("c", now=11.0)[0]
    assert not lim.check("c", now=11.0)[0]


def test_retry_after_points_at_when_a_slot_frees():
    lim = limits.SlidingWindowLimiter(max_requests=1, window_seconds=30)
    lim.check("c", now=100.0)
    allowed, retry_after = lim.check("c", now=110.0)
    assert not allowed
    # The one hit was at 100 and ages out at 130, i.e. 20s from now.
    assert retry_after == pytest.approx(20.0)


def test_idle_clients_do_not_accumulate_for_ever():
    """The client key can come from a spoofable header, so a map that grows
    once per distinct key and never shrinks is a memory hole, not untidiness."""
    lim = limits.SlidingWindowLimiter(max_requests=1, window_seconds=1, sweep_threshold=100)
    for i in range(1000):
        lim.check(f"client-{i}", now=float(i))
    # Each client is idle by the time the next arrives, so almost nothing should
    # still be tracked — certainly not one entry per caller ever seen.
    assert len(lim._hits) <= 101, f"tracking {len(lim._hits)} idle clients"


class _Req:
    def __init__(self, headers=None, host="10.0.0.1"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})()


def test_client_key_prefers_the_forwarded_address():
    """Behind a load balancer the socket address is the balancer, so every
    caller would share one budget without this."""
    req = _Req({"x-forwarded-for": "203.0.113.7, 70.41.3.18"}, host="10.0.0.5")
    assert limits.client_key(req) == "203.0.113.7"


def test_client_key_falls_back_to_the_socket_address():
    assert limits.client_key(_Req(host="198.51.100.9")) == "198.51.100.9"


def test_client_key_survives_a_missing_client():
    req = _Req()
    req.client = None
    assert limits.client_key(req) == "unknown"


# --- endpoint level -----------------------------------------------------------
#
# These drive the real ASGI app so the 429 path is covered end to end, but they
# stub the model and index rather than running startup: loading bge-small takes
# ~20s and would download 128 MB in CI to test something that has nothing to do
# with embeddings.


class _FakeIndex:
    def search(self, vector, k):
        return [], []


class _FakeModel:
    def encode(self, text, normalize_embeddings=True):
        import numpy as np

        v = np.ones(384, dtype=np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from api import main

    main._state["model"] = _FakeModel()
    main._state["index"] = _FakeIndex()
    main._limiter.reset()
    monkeypatch.setattr(
        "api.expand.search_vector",
        lambda model, q, alpha=0.0: (_FakeModel().encode(q), None, None),
    )
    # Stub the stance layer too. Left real it tries an LLM backend on every
    # request and waits for it to fail, which turns a test about rate limiting
    # into 20 timeouts — slow, and dependent on whether a `claude` binary
    # happens to be on PATH.
    monkeypatch.setattr("api.stance.classify", lambda claim, passages: ({}, None))
    # And claim resolution, for the same reason: a short query like "claim 5"
    # reads as a bare topic, which sends it to the LLM to be turned into a
    # canonical claim. Twenty of those is twenty backend timeouts.
    monkeypatch.setattr("api.main.resolve_claim", lambda q: (q, False, None))
    return TestClient(main.app)


def test_endpoint_refuses_past_the_budget(client):
    from api import limits as L

    codes = [client.get(f"/search?q=claim+{i}").status_code
             for i in range(L.MAX_REQUESTS_PER_WINDOW + 3)]
    assert codes[:L.MAX_REQUESTS_PER_WINDOW] == [200] * L.MAX_REQUESTS_PER_WINDOW
    assert codes[L.MAX_REQUESTS_PER_WINDOW:] == [429, 429, 429]


def test_the_429_tells_the_caller_when_to_come_back(client):
    from api import limits as L

    for i in range(L.MAX_REQUESTS_PER_WINDOW):
        client.get(f"/search?q=claim+{i}")
    r = client.get("/search?q=one+too+many")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_over_long_queries_are_rejected_before_any_work(client):
    from api import limits as L

    r = client.get("/search?q=" + "a" * (L.MAX_QUERY_CHARS + 1))
    assert r.status_code == 422  # FastAPI validation, no model call made
