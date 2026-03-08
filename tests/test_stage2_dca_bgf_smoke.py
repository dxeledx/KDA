import numpy as np
from pyriemann.utils.mean import mean_riemann

from src.alignment.behavior_feedback import BehaviorGuidedFeedback
from src.alignment.conditional import LinearConditionalWeight
from src.alignment.dca_bgf import DCABGF
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.context import ContextComputer


def _balanced_labels(n_trials: int, n_classes: int = 4) -> np.ndarray:
    base = np.repeat(np.arange(n_classes), n_trials // n_classes)
    remainder = n_trials - len(base)
    if remainder > 0:
        base = np.concatenate([base, np.arange(remainder)])
    rng = np.random.RandomState(0)
    rng.shuffle(base)
    return base.astype(np.int64)


def test_dca_bgf_predict_online_smoke():
    rng = np.random.RandomState(0)
    n_source, n_target, n_channels, n_samples = 80, 20, 22, 200
    X_source = rng.randn(n_source, n_channels, n_samples)
    y_source = _balanced_labels(n_source)
    X_target = rng.randn(n_target, n_channels, n_samples)
    y_target = _balanced_labels(n_target)

    csp = CSP(n_components=6)
    feats_source = csp.fit_transform(X_source, y_source)

    lda = LDA().fit(feats_source, y_source)
    alignment_matrix = np.eye(n_channels, dtype=np.float64)

    source_covariances = compute_covariances(X_source)
    ctx = ContextComputer(
        source_mean=feats_source.mean(axis=0),
        feature_names=["d_src", "d_tgt", "sigma_recent", "d_geo"],
        source_covariance=mean_riemann(source_covariances),
        normalize=True,
    )
    ctx.fit_normalizer(feats_source, covariance_stream=source_covariances)
    conditional = LinearConditionalWeight(weights=[1.2, 0.6, 0.3, 1.2], ema_smooth_alpha=0.2)
    feedback = BehaviorGuidedFeedback(window_size=10)

    dca = DCABGF(
        csp=csp,
        lda=lda,
        alignment_matrix=alignment_matrix,
        context_computer=ctx,
        conditional_weight=conditional,
        behavior_feedback=feedback,
        use_feedback=True,
    )

    y_pred, details = dca.predict_online(
        X_target, y_true=y_target, return_details=True, return_features=True
    )

    assert y_pred.shape == (n_target,)
    assert details["w"].shape == (n_target,)
    assert details["conf"].shape == (n_target,)
    assert details["entropy"].shape == (n_target,)
    assert details["context"].shape == (n_target, 4)
    assert details["features_final"].shape[0] == n_target
    assert np.isfinite(details["features_final"]).all()
