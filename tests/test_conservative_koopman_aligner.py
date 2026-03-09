import numpy as np
import pytest


def _balanced_labels(n: int, n_classes: int = 4) -> np.ndarray:
    labels = np.arange(n, dtype=np.int64) % n_classes
    return labels


def test_conservative_aligner_zero_step_is_identity_like():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    rng = np.random.RandomState(0)
    source = rng.randn(40, 12)
    target = source + 1.5
    y_source = _balanced_labels(len(source))

    aligner = KoopmanConservativeResidualAligner(
        residual_rank=4,
        basis_k=4,
        lambda_dyn=0.0,
        max_iter=0,
    ).fit(source, target, y_source=y_source)

    transformed_source = aligner.transform_source(source)
    transformed_target = aligner.transform_target(target)

    assert np.allclose(aligner.residual_matrix_, np.zeros((4, 4)))
    assert np.allclose(transformed_source, source)
    assert np.allclose(transformed_target.mean(axis=0), source.mean(axis=0))


def test_conservative_aligner_outputs_finite_with_nonzero_optimization():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    rng = np.random.RandomState(1)
    source = rng.randn(48, 10)
    target = source * 1.2 + 0.8
    y_source = _balanced_labels(len(source))

    aligner = KoopmanConservativeResidualAligner(
        residual_rank=3,
        basis_k=6,
        lambda_dyn=0.0,
        max_iter=5,
    ).fit(source, target, y_source=y_source)

    source_out = aligner.transform_source(source[:5])
    target_out = aligner.transform_target(target[:7])

    assert source_out.shape == (5, 10)
    assert target_out.shape == (7, 10)
    assert np.isfinite(source_out).all()
    assert np.isfinite(target_out).all()


def test_conservative_aligner_requires_source_block_lengths_when_dyn_enabled():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    rng = np.random.RandomState(2)
    source = rng.randn(32, 8)
    target = rng.randn(16, 8)
    y_source = _balanced_labels(len(source))

    with pytest.raises(ValueError, match="source_block_lengths"):
        KoopmanConservativeResidualAligner(
            residual_rank=3,
            basis_k=4,
            lambda_dyn=1.0,
        ).fit(source, target, y_source=y_source, source_block_lengths=None)


def test_rank_targets_follow_same_order_as_rbid():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    behavior_scores = np.array([0.2, 0.8, 0.5], dtype=np.float64)
    rank_targets = KoopmanConservativeResidualAligner.compute_rank_targets(behavior_scores)

    assert rank_targets.shape == behavior_scores.shape
    assert np.all(rank_targets >= 0.0)
    assert np.all(rank_targets <= 1.0)
    assert rank_targets[1] > rank_targets[2] > rank_targets[0]


def test_soft_ranks_preserve_score_order_and_range():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    scores = np.array([0.1, 0.8, 0.4], dtype=np.float64)
    soft_ranks = KoopmanConservativeResidualAligner.compute_soft_ranks(scores, tau=0.1)

    assert soft_ranks.shape == scores.shape
    assert np.all(soft_ranks >= 0.0)
    assert np.all(soft_ranks <= 1.0)
    assert soft_ranks[1] > soft_ranks[2] > soft_ranks[0]


def test_soft_rbid_loss_prefers_matching_rank_targets():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    behavior_scores = np.array([0.2, 0.8, 0.5], dtype=np.float64)
    good_scores = np.array([0.1, 0.7, 0.4], dtype=np.float64)
    bad_scores = np.array([0.8, 0.2, 0.4], dtype=np.float64)

    good = KoopmanConservativeResidualAligner.compute_soft_rbid_loss(
        behavior_scores=behavior_scores,
        similarity_scores=good_scores,
        tau=0.1,
        huber_delta=0.1,
    )
    bad = KoopmanConservativeResidualAligner.compute_soft_rbid_loss(
        behavior_scores=behavior_scores,
        similarity_scores=bad_scores,
        tau=0.1,
        huber_delta=0.1,
    )

    assert good < bad


def test_tail_weighted_soft_rbid_penalizes_low_behavior_rank_mismatch_more():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    behavior_scores = np.array([0.1, 0.8, 0.5, 0.2], dtype=np.float64)
    low_tail_bad = np.array([0.9, 0.8, 0.5, 0.2], dtype=np.float64)
    mid_bad = np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float64)

    low_tail_loss = KoopmanConservativeResidualAligner.compute_soft_rbid_loss(
        behavior_scores=behavior_scores,
        similarity_scores=low_tail_bad,
        tau=0.1,
        huber_delta=0.1,
        tail_weight=2.0,
        tail_quantile=0.25,
    )
    mid_loss = KoopmanConservativeResidualAligner.compute_soft_rbid_loss(
        behavior_scores=behavior_scores,
        similarity_scores=mid_bad,
        tau=0.1,
        huber_delta=0.1,
        tail_weight=2.0,
        tail_quantile=0.25,
    )

    assert low_tail_loss > mid_loss
