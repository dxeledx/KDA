#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.euclidean import EuclideanAlignment, apply_alignment
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import cka
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("validate_stage1")


def validate_implementation() -> None:
    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))

    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))

    logger.info("=" * 50)
    logger.info("Validating data loading...")
    X_train, y_train, X_test, y_test = loader.get_train_test_split(subject_id=1)

    assert X_train.shape[1] == 22, f"Wrong channel count: {X_train.shape}"
    assert X_train.shape[2] == 750, f"Wrong sample count: {X_train.shape}"
    assert X_test.shape == X_train.shape, f"Train/test shape mismatch: {X_train.shape} vs {X_test.shape}"
    assert y_train.shape == (288,), f"Wrong y_train shape: {y_train.shape}"
    assert set(np.unique(y_train)) == {0, 1, 2, 3}, f"Wrong label values: {np.unique(y_train)}"

    logger.info("✓ Data loading OK: train=%s test=%s", X_train.shape, X_test.shape)

    logger.info("=" * 50)
    logger.info("Validating CSP...")
    csp = CSP(**model_cfg["csp"])
    X_train_p = pre.fit(X_train, y_train).transform(X_train)
    X_test_p = pre.transform(X_test)

    features_train = csp.fit_transform(X_train_p, y_train)
    assert features_train.shape == (288, 24), f"Wrong feature shape: {features_train.shape}"
    assert not np.isnan(features_train).any(), "Features contain NaN!"
    logger.info("✓ CSP OK: features_train=%s", features_train.shape)

    logger.info("=" * 50)
    logger.info("Validating LDA...")
    lda = LDA(**model_cfg["lda"])
    lda.fit(features_train, y_train)
    train_acc = lda.score(features_train, y_train)
    assert train_acc > 0.7, f"Train accuracy too low: {train_acc}"

    features_test = csp.transform(X_test_p)
    test_acc = lda.score(features_test, y_test)
    assert test_acc > 0.6, f"Test accuracy too low: {test_acc}"
    logger.info("✓ LDA OK (train=%.3f, test=%.3f)", train_acc, test_acc)

    logger.info("=" * 50)
    logger.info("Validating alignment...")
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))
    C_train = compute_covariances(X_train_p, eps=cov_eps)
    C_test = compute_covariances(X_test_p, eps=cov_eps)

    ea = EuclideanAlignment(eps=cov_eps).fit(C_train)
    A = ea.compute_matrix(C_test)
    X_test_aligned = apply_alignment(X_test_p, A)
    assert X_test_aligned.shape == X_test_p.shape
    assert not np.isnan(X_test_aligned).any(), "Aligned data contains NaN!"

    # Alignment effectiveness (domain mean covariance distance should decrease)
    C_source_mean = np.mean(C_train, axis=0)
    C_target_mean = np.mean(C_test, axis=0)
    dist_before = float(np.linalg.norm(C_source_mean - C_target_mean, ord="fro"))
    C_target_aligned_mean = A @ C_target_mean @ A.T
    dist_after = float(np.linalg.norm(C_source_mean - C_target_aligned_mean, ord="fro"))
    logger.info(
        "Alignment distance (Frobenius): before=%.4f after=%.4f (%.1f%% reduction)",
        dist_before,
        dist_after,
        100.0 * (dist_before - dist_after) / max(dist_before, 1e-12),
    )
    assert dist_after <= dist_before + 1e-6, "Alignment did not reduce domain distance!"
    logger.info("✓ Alignment OK")

    logger.info("=" * 50)
    logger.info("Validating CKA...")
    f1 = csp.transform(X_test_p[:100])
    f2 = csp.transform(X_test_p[100:200])
    cka_val = cka(f1, f2)
    assert 0.0 <= cka_val <= 1.0, f"CKA out of range: {cka_val}"
    cka_self = cka(f1, f1)
    assert np.isclose(cka_self, 1.0, atol=1e-3), f"CKA self != 1: {cka_self}"
    logger.info("✓ CKA OK (value=%.3f)", cka_val)

    logger.info("=" * 50)
    logger.info("All checks passed!")


if __name__ == "__main__":
    validate_implementation()
