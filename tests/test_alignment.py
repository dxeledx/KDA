import numpy as np

from src.alignment.euclidean import EuclideanAlignment, apply_alignment
from src.alignment.riemannian import RiemannianAlignment
from src.features.covariance import compute_covariances


def test_ea_alignment_shapes():
    rng = np.random.RandomState(0)
    X_src = rng.randn(50, 22, 200)
    X_tgt = rng.randn(60, 22, 200)

    C_src = compute_covariances(X_src, eps=1e-6)
    C_tgt = compute_covariances(X_tgt, eps=1e-6)

    ea = EuclideanAlignment(eps=1e-6).fit(C_src)
    A = ea.compute_matrix(C_tgt)
    assert A.shape == (22, 22)
    assert np.isfinite(A).all()

    X_aligned = apply_alignment(X_tgt, A)
    assert X_aligned.shape == X_tgt.shape
    assert np.isfinite(X_aligned).all()


def test_ra_alignment_shapes():
    rng = np.random.RandomState(0)
    X_src = rng.randn(50, 22, 200)
    X_tgt = rng.randn(60, 22, 200)

    C_src = compute_covariances(X_src, eps=1e-6)
    C_tgt = compute_covariances(X_tgt, eps=1e-6)

    ra = RiemannianAlignment(eps=1e-6).fit(C_src)
    A = ra.compute_matrix(C_tgt)
    assert A.shape == (22, 22)
    assert np.isfinite(A).all()

