import numpy as np
import pytest

from src.utils.context import ContextComputer


def test_context_computer_shape_and_state():
    rng = np.random.RandomState(0)
    d = 24
    source_mean = np.zeros(d, dtype=np.float64)
    ctx = ContextComputer(
        source_mean=source_mean, context_dim=3, ema_alpha=0.2, recent_window=5, normalize=False
    )

    history = []
    x1 = rng.randn(d)
    c1 = ctx.compute(x1, history)
    assert c1.shape == (3,)
    assert np.isfinite(c1).all()
    assert ctx.target_mean is not None

    history.append({"x": x1})
    x2 = rng.randn(d)
    c2 = ctx.compute(x2, history)
    assert c2.shape == (3,)
    assert np.isfinite(c2).all()
    assert not np.allclose(c1, c2)


def test_context_normalizer():
    rng = np.random.RandomState(0)
    n, d = 50, 24
    features = rng.randn(n, d)
    source_mean = features.mean(axis=0)
    ctx = ContextComputer(source_mean=source_mean, normalize=True)
    stats = ctx.fit_normalizer(features)
    assert "mean" in stats and "std" in stats
    assert stats["mean"].shape == (3,)
    assert stats["std"].shape == (3,)

    history = []
    c = ctx.compute(features[0], history)
    assert c.shape == (3,)
    assert np.isfinite(c).all()


def test_context_computer_supports_d_geo_feature():
    rng = np.random.RandomState(0)
    d = 6
    source_mean = np.zeros(d, dtype=np.float64)
    source_covariance = np.eye(3, dtype=np.float64)

    ctx = ContextComputer(
        source_mean=source_mean,
        feature_names=["d_src", "d_tgt", "sigma_recent", "d_geo"],
        source_covariance=source_covariance,
        normalize=False,
    )

    x_t = rng.randn(d)
    trial_cov = 2.0 * np.eye(3, dtype=np.float64)
    c_t = ctx.compute(x_t, history=[], trial_cov=trial_cov)

    assert c_t.shape == (4,)
    assert np.isfinite(c_t).all()
    assert c_t[-1] > 0.0


def test_context_normalizer_with_d_geo_requires_covariances():
    rng = np.random.RandomState(0)
    features = rng.randn(10, 6)
    ctx = ContextComputer(
        source_mean=features.mean(axis=0),
        feature_names=["d_src", "d_tgt", "sigma_recent", "d_geo"],
        source_covariance=np.eye(3, dtype=np.float64),
        normalize=True,
    )

    with pytest.raises(ValueError, match="covariance_stream"):
        ctx.fit_normalizer(features)
