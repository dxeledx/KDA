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
