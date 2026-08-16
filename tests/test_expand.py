"""Query expansion: the blend maths, the cache, and the no-backend fallback."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import expand


def test_blend_stays_on_the_unit_sphere():
    """Cosine similarity is what the index scores on, so a blended query that
    is not unit length would silently change every score."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=384).astype(np.float32)
        b = rng.normal(size=384).astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        for alpha in (0.0, 0.3, 0.5, 1.0):
            assert np.isclose(np.linalg.norm(expand.blend(a, b, alpha)), 1.0, atol=1e-5)


def test_blend_endpoints_are_the_inputs():
    a = np.zeros(4, dtype=np.float32); a[0] = 1.0
    b = np.zeros(4, dtype=np.float32); b[1] = 1.0
    assert np.allclose(expand.blend(a, b, 0.0), a)
    assert np.allclose(expand.blend(a, b, 1.0), b)
    # Halfway is equidistant from both, which is the property that makes alpha
    # readable as "how far toward the hypothetical".
    mid = expand.blend(a, b, 0.5)
    assert np.isclose(float(mid @ a), float(mid @ b))


def test_blend_moves_toward_the_hypothetical_monotonically():
    a = np.zeros(4, dtype=np.float32); a[0] = 1.0
    b = np.zeros(4, dtype=np.float32); b[1] = 1.0
    sims = [float(expand.blend(a, b, al) @ b) for al in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert sims == sorted(sims)


def test_degenerate_blend_falls_back_to_the_query():
    """Opposite vectors sum to zero; retrieval must not divide by it."""
    a = np.zeros(4, dtype=np.float32); a[0] = 1.0
    out = expand.blend(a, -a, 0.5)
    assert np.allclose(out, a)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(expand, "DB_PATH", tmp_path / "t.db")
    assert expand.cached("a novel claim") is None
    expand.store("a novel claim", "  a hypothetical passage  ")
    assert expand.cached("a novel claim") == "a hypothetical passage"


def test_search_vector_without_a_backend_returns_the_plain_query(tmp_path, monkeypatch):
    """No LLM reachable is a normal state, not an error: search still works,
    it just stops being expanded."""
    monkeypatch.setattr(expand, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(expand, "_sdk_available", lambda: False)

    def boom(_prompt):
        raise RuntimeError("no backend")

    monkeypatch.setattr(expand, "_classify_via_cli_text", boom)

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            v = np.ones(4, dtype=np.float32)
            return v / np.linalg.norm(v)

    vec, hypothetical, error = expand.search_vector(FakeModel(), "some claim")
    assert hypothetical is None
    assert error and "unavailable" in error
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_alpha_zero_skips_the_backend_entirely(tmp_path, monkeypatch):
    monkeypatch.setattr(expand, "DB_PATH", tmp_path / "t.db")

    def should_not_run(_claim):
        raise AssertionError("alpha=0 must not call the expansion backend")

    monkeypatch.setattr(expand, "expand", should_not_run)

    class FakeModel:
        def encode(self, text, normalize_embeddings=True):
            v = np.arange(4, dtype=np.float32) + 1
            return v / np.linalg.norm(v)

    vec, hypothetical, error = expand.search_vector(FakeModel(), "claim", alpha=0.0)
    assert hypothetical is None and error is None
    assert np.isclose(np.linalg.norm(vec), 1.0)
