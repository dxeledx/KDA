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


def test_source_context_normalizer_depends_only_on_source_covariances():
    from experiments.trial_dynamic_gate_exp_b1 import fit_source_context_normalizer

    rng = np.random.RandomState(0)
    source_covariances = compute_covariances(rng.randn(20, 8, 64), eps=1.0e-6)
    source_mean_covariance = source_covariances.mean(axis=0)

    stats_a = fit_source_context_normalizer(
        source_covariances=source_covariances,
        source_mean_covariance=source_mean_covariance,
        ema_alpha=0.1,
        recent_window=5,
    )
    stats_b = fit_source_context_normalizer(
        source_covariances=source_covariances,
        source_mean_covariance=source_mean_covariance,
        ema_alpha=0.1,
        recent_window=5,
    )

    assert np.allclose(stats_a["mean"], stats_b["mean"])
    assert np.allclose(stats_a["std"], stats_b["std"])


def test_normalized_context_reduces_scale_disparity():
    from experiments.trial_dynamic_gate_exp_b1 import normalize_trial_geometry_context

    raw = np.array(
        [
            [10.0, 1.0, 0.1],
            [12.0, 1.2, 0.2],
            [14.0, 1.4, 0.3],
            [16.0, 1.6, 0.4],
        ],
        dtype=np.float64,
    )
    stats = {"mean": raw.mean(axis=0), "std": raw.std(axis=0)}
    norm = normalize_trial_geometry_context(raw, stats)

    assert np.isfinite(norm).all()
    assert np.allclose(norm.mean(axis=0), 0.0, atol=1.0e-7)
    assert np.allclose(norm.std(axis=0), 1.0, atol=1.0e-6)


def test_dynamic_exp_b1_returns_raw_and_normalized_contexts():
    from experiments.trial_dynamic_gate_exp_b1 import (
        compute_trial_geometry_context,
        fit_source_context_normalizer,
        normalize_trial_geometry_context,
        predict_dynamic_covariance_space_normalized,
    )

    rng = np.random.RandomState(0)
    n_source, n_target, n_channels, n_samples = 64, 24, 8, 64
    X_source = rng.randn(n_source, n_channels, n_samples)
    y_source = _balanced_labels(n_source)
    X_target = rng.randn(n_target, n_channels, n_samples)

    csp = CSP(n_components=4)
    features_source = csp.fit_transform(X_source, y_source)
    lda = LDA().fit(features_source, y_source)

    source_covariances = compute_covariances(X_source, eps=1.0e-6)
    source_mean_covariance = source_covariances.mean(axis=0)
    stats = fit_source_context_normalizer(
        source_covariances=source_covariances,
        source_mean_covariance=source_mean_covariance,
        ema_alpha=0.1,
        recent_window=5,
    )

    cov_raw = compute_covariances(X_target, eps=1.0e-6)
    cov_ra = cov_raw[::-1].copy()
    context_raw = compute_trial_geometry_context(
        covariances=cov_raw,
        source_mean_covariance=source_mean_covariance,
        ema_alpha=0.1,
        recent_window=5,
    )
    context_norm = normalize_trial_geometry_context(context_raw, stats)
    gate = LinearConditionalWeight(weights=[1.2, 0.6, 0.3], bias=0.0, temperature=1.0, ema_smooth_alpha=0.2)

    y_pred, details = predict_dynamic_covariance_space_normalized(
        csp=csp,
        lda=lda,
        cov_raw=cov_raw,
        cov_ra=cov_ra,
        gate=gate,
        context_raw=context_raw,
        context_norm=context_norm,
    )

    assert y_pred.shape == (n_target,)
    assert details["context_raw"].shape == (n_target, 3)
    assert details["context_norm"].shape == (n_target, 3)
    assert details["w"].shape == (n_target,)
    assert np.isfinite(details["context_norm"]).all()
