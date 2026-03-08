import numpy as np

from src.evaluation.metrics import cka


def test_cka_self_is_one():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 24)
    val = cka(X, X)
    assert np.isclose(val, 1.0, atol=1e-6)


def test_cka_range():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 24)
    Y = rng.randn(200, 24)
    val = cka(X, Y)
    assert 0.0 <= val <= 1.0 + 1e-9

