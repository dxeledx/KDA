import numpy as np


def test_koopman_feature_projector_output_dimension():
    from src.alignment.koopman_alignment import KoopmanFeatureProjector
    from src.features.covariance import compute_covariances

    rng = np.random.RandomState(0)
    X = rng.randn(40, 8, 64)
    covariances = compute_covariances(X, eps=1.0e-6)

    projector = KoopmanFeatureProjector(pca_rank=6).fit(covariances)
    features = projector.transform(covariances)

    assert features.shape == (40, 13)
    assert np.isfinite(features).all()


def test_koopman_affine_aligner_reduces_mean_distance():
    from src.alignment.koopman_alignment import KoopmanAffineAligner

    rng = np.random.RandomState(0)
    source = rng.randn(80, 10)
    target = 2.0 * source + 3.0

    aligner = KoopmanAffineAligner().fit(source, target)
    aligned = aligner.transform(target)

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    aligned_mean = aligned.mean(axis=0)

    before = np.linalg.norm(target_mean - source_mean)
    after = np.linalg.norm(aligned_mean - source_mean)
    assert after < before


def test_koopman_affine_aligner_reduces_covariance_gap():
    from src.alignment.koopman_alignment import KoopmanAffineAligner

    rng = np.random.RandomState(0)
    source = rng.randn(120, 8)
    target = source @ np.diag(np.linspace(1.5, 2.5, 8)) + 0.5

    aligner = KoopmanAffineAligner().fit(source, target)
    aligned = aligner.transform(target)

    source_cov = np.cov(source, rowvar=False)
    target_cov = np.cov(target, rowvar=False)
    aligned_cov = np.cov(aligned, rowvar=False)

    before = np.linalg.norm(target_cov - source_cov, ord="fro")
    after = np.linalg.norm(aligned_cov - source_cov, ord="fro")
    assert after < before
