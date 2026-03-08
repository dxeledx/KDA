import numpy as np

from src.features.csp import CSP


def _balanced_labels(n_trials: int, n_classes: int = 4) -> np.ndarray:
    base = np.repeat(np.arange(n_classes), n_trials // n_classes)
    remainder = n_trials - len(base)
    if remainder > 0:
        base = np.concatenate([base, np.arange(remainder)])
    rng = np.random.RandomState(0)
    rng.shuffle(base)
    return base.astype(np.int64)


def test_csp_shape():
    n_trials, n_channels, n_samples = 120, 22, 750
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_samples)
    y = _balanced_labels(n_trials)

    csp = CSP(n_components=6)
    features = csp.fit_transform(X, y)
    assert features.shape == (n_trials, 24)


def test_csp_no_nan():
    n_trials, n_channels, n_samples = 120, 22, 750
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_samples)
    y = _balanced_labels(n_trials)

    csp = CSP(n_components=6)
    features = csp.fit_transform(X, y)
    assert not np.isnan(features).any()


def test_csp_deterministic():
    n_trials, n_channels, n_samples = 120, 22, 750
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_samples)
    y = _balanced_labels(n_trials)

    csp1 = CSP(n_components=6)
    f1 = csp1.fit_transform(X, y)

    csp2 = CSP(n_components=6)
    f2 = csp2.fit_transform(X, y)

    assert np.allclose(f1, f2)

