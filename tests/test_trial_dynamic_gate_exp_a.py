import numpy as np

from src.features.covariance import compute_covariances
from src.features.csp import CSP


def _balanced_labels(n_trials: int, n_classes: int = 4) -> np.ndarray:
    base = np.repeat(np.arange(n_classes), n_trials // n_classes)
    remainder = n_trials - len(base)
    if remainder > 0:
        base = np.concatenate([base, np.arange(remainder)])
    rng = np.random.RandomState(0)
    rng.shuffle(base)
    return base.astype(np.int64)


def test_csp_transform_covariances_matches_signal_transform():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 8, 64)
    y = _balanced_labels(40)

    csp = CSP(n_components=4)
    csp.fit(X, y)

    signal_features = csp.transform(X)
    covariances = compute_covariances(X, eps=1.0e-6)
    covariance_features = csp.transform_covariances(covariances)

    assert covariance_features.shape == signal_features.shape
    assert np.allclose(covariance_features, signal_features, atol=1.0e-6)


def test_covariance_space_partial_alignment_interpolates_between_endpoints():
    rng = np.random.RandomState(0)
    X = rng.randn(32, 8, 64)
    y = _balanced_labels(32)
    csp = CSP(n_components=4)
    csp.fit(X, y)

    covariances = compute_covariances(X, eps=1.0e-6)
    cov_raw = covariances[0]
    cov_ra = covariances[1]

    feat_raw = csp.transform_covariances(cov_raw[None, ...])[0]
    feat_ra = csp.transform_covariances(cov_ra[None, ...])[0]

    w = 0.5
    cov_mix = (1.0 - w) * cov_raw + w * cov_ra
    feat_mix = csp.transform_covariances(cov_mix[None, ...])[0]

    assert np.isfinite(feat_mix).all()
    assert feat_mix.shape == feat_raw.shape
    assert not np.allclose(feat_mix, feat_raw)
    assert not np.allclose(feat_mix, feat_ra)
