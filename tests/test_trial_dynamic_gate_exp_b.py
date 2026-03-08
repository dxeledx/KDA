import numpy as np

from src.alignment.conditional import LinearConditionalWeight
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA


def _balanced_labels(n_trials: int, n_classes: int = 4) -> np.ndarray:
    base = np.repeat(np.arange(n_classes), n_trials // n_classes)
    remainder = n_trials - len(base)
    if remainder > 0:
        base = np.concatenate([base, np.arange(remainder)])
    rng = np.random.RandomState(0)
    rng.shuffle(base)
    return base.astype(np.int64)


def test_compute_trial_geometry_context_has_expected_shape_and_warmup():
    from experiments.trial_dynamic_gate_exp_b import compute_trial_geometry_context

    rng = np.random.RandomState(0)
    X = rng.randn(10, 8, 64)
    covs = compute_covariances(X, eps=1.0e-6)
    source_mean_cov = covs.mean(axis=0)

    context = compute_trial_geometry_context(
        covariances=covs,
        source_mean_covariance=source_mean_cov,
        ema_alpha=0.1,
        recent_window=5,
    )

    assert context.shape == (10, 3)
    assert np.isfinite(context).all()
    assert np.allclose(context[:5, 2], 0.0)
    assert np.any(context[5:, 2] > 0.0)


def test_dynamic_exp_b_uses_nonconstant_w_and_tracks_context():
    from experiments.trial_dynamic_gate_exp_b import (
        compute_trial_geometry_context,
        predict_dynamic_covariance_space,
    )

    rng = np.random.RandomState(0)
    n_source, n_target, n_channels, n_samples = 64, 24, 8, 64
    X_source = rng.randn(n_source, n_channels, n_samples)
    y_source = _balanced_labels(n_source)
    X_target = rng.randn(n_target, n_channels, n_samples)

    csp = CSP(n_components=4)
    features_source = csp.fit_transform(X_source, y_source)
    lda = LDA().fit(features_source, y_source)

    cov_raw = compute_covariances(X_target, eps=1.0e-6)
    cov_ra = cov_raw[::-1].copy()
    context = compute_trial_geometry_context(
        covariances=cov_raw,
        source_mean_covariance=compute_covariances(X_source, eps=1.0e-6).mean(axis=0),
        ema_alpha=0.1,
        recent_window=5,
    )
    gate = LinearConditionalWeight(weights=[1.2, 0.6, 0.3], bias=0.5, ema_smooth_alpha=0.0)

    y_pred, details = predict_dynamic_covariance_space(
        csp=csp,
        lda=lda,
        cov_raw=cov_raw,
        cov_ra=cov_ra,
        gate=gate,
        context=context,
    )

    assert y_pred.shape == (n_target,)
    assert details["context"].shape == (n_target, 3)
    assert details["w"].shape == (n_target,)
    assert np.isfinite(details["w"]).all()
    assert details["w"].std() > 0.0
    assert np.corrcoef(details["w"], details["d_tgt"])[0, 1] > 0.0
