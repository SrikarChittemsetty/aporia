import numpy as np
import pytest

from index.vector_index import NumpyIndex, PyHnswIndex


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = [i * 3 for i in range(500)]  # non-contiguous external ids
    return x, ids


def test_numpy_index_exact_self_search(data):
    x, ids = data
    index = NumpyIndex()
    index.build(x, ids)
    got, scores = index.search(x[7], k=1)
    assert got[0] == ids[7]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_pyhnsw_recall_against_brute_force(data):
    x, ids = data
    brute = NumpyIndex(); brute.build(x, ids)
    ours = PyHnswIndex(m=8, ef_construction=100, ef_search=50)
    ours.build(x, ids)

    rng = np.random.default_rng(2)
    hits = total = 0
    for q in rng.standard_normal((20, 32)).astype(np.float32):
        true_ids, _ = brute.search(q, 10)
        got_ids, _ = ours.search(q, 10)
        hits += len(set(true_ids) & set(got_ids))
        total += 10
    assert hits / total >= 0.9


def test_indexes_roundtrip_through_disk(tmp_path, data):
    x, ids = data
    for cls in (NumpyIndex, PyHnswIndex):
        d = tmp_path / cls.__name__
        index = cls()
        index.build(x, ids)
        before, _ = index.search(x[3], k=5)
        index.save(d)
        loaded = cls.load(d)
        after, _ = loaded.search(x[3], k=5)
        assert before == after
