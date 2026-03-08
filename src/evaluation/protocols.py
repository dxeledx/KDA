from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann

from src.alignment.euclidean import apply_alignment, compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.covariance import compute_covariances, mean_covariance
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.logger import get_logger


_LOGGER = get_logger(__name__)

Method = Literal["noalign", "ea", "ra"]


def _domain_mean_covariances(
    X: np.ndarray, method: Method, eps: float
) -> np.ndarray:  # noqa: N803
    covs = compute_covariances(X, eps=eps)
    if method == "ea":
        return mean_covariance(covs)
    if method == "ra":
        return mean_riemann(covs)
    raise ValueError(f"Unsupported mean-cov method: {method}")


def train_csp_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    csp_kwargs: Dict[str, Any],
    lda_kwargs: Dict[str, Any],
) -> Tuple[CSP, LDA]:
    csp = CSP(**csp_kwargs)
    features_train = csp.fit_transform(X_train, y_train)
    lda = LDA(**lda_kwargs)
    lda.fit(features_train, y_train)
    return csp, lda


def evaluate_within_subject(
    loader: BCIDataLoader,
    subjects: List[int],
    preprocessor: Preprocessor,
    csp_kwargs: Dict[str, Any],
    lda_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        X_train, y_train, X_test, y_test = loader.get_train_test_split(subject)
        X_train = preprocessor.fit(X_train, y_train).transform(X_train)
        X_test = preprocessor.transform(X_test)

        csp, lda = train_csp_lda(X_train, y_train, csp_kwargs, lda_kwargs)
        train_acc = lda.score(csp.transform(X_train), y_train)
        y_pred = lda.predict(csp.transform(X_test))
        metrics = compute_metrics(y_test, y_pred)
        rows.append(
            {
                "subject": subject,
                "train_accuracy": float(train_acc),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def evaluate_loso(
    loader: BCIDataLoader,
    subjects: List[int],
    preprocessor: Preprocessor,
    csp_kwargs: Dict[str, Any],
    lda_kwargs: Dict[str, Any],
    method: Method,
    cov_eps: float,
) -> pd.DataFrame:
    rows = []
    for target in subjects:
        X_train_list, y_train_list = [], []
        for s in subjects:
            if s == target:
                continue
            X_s, y_s = loader.load_subject(s, split="train")
            X_train_list.append(X_s)
            y_train_list.append(y_s)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_t_train, _y_t_train = loader.load_subject(target, split="train")
        X_test, y_test = loader.load_subject(target, split="test")

        X_train = preprocessor.fit(X_train, y_train).transform(X_train)
        X_t_train = preprocessor.transform(X_t_train)
        X_test = preprocessor.transform(X_test)

        if method in ("ea", "ra"):
            C_source = _domain_mean_covariances(X_train, method=method, eps=cov_eps)
            C_target = _domain_mean_covariances(X_t_train, method=method, eps=cov_eps)
            A = compute_alignment_matrix(C_source, C_target, eps=cov_eps)
            X_test_eval = apply_alignment(X_test, A)
        else:
            X_test_eval = X_test

        csp, lda = train_csp_lda(X_train, y_train, csp_kwargs, lda_kwargs)
        y_pred = lda.predict(csp.transform(X_test_eval))
        metrics = compute_metrics(y_test, y_pred)
        rows.append({"target_subject": target, **metrics})

        _LOGGER.info("LOSO target=%s metrics=%s", target, metrics)

    return pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)


def evaluate_pairwise_transfer(
    loader: BCIDataLoader,
    subjects: List[int],
    preprocessor: Preprocessor,
    csp_kwargs: Dict[str, Any],
    lda_kwargs: Dict[str, Any],
    method: Method,
    cov_eps: float,
) -> Dict[str, np.ndarray]:
    n = len(subjects)
    acc = np.zeros((n, n), dtype=np.float64)
    kappa = np.zeros((n, n), dtype=np.float64)
    f1 = np.zeros((n, n), dtype=np.float64)

    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    domain_mean = None
    if method in ("ea", "ra"):
        domain_mean = {}
        for sid in subjects:
            X_train, y_train = loader.load_subject(sid, split="train")
            X_train = preprocessor.fit(X_train, y_train).transform(X_train)
            domain_mean[sid] = _domain_mean_covariances(X_train, method=method, eps=cov_eps)

    for source in subjects:
        src_idx = subject_to_idx[source]
        X_source_train, y_source_train = loader.load_subject(source, split="train")
        X_source_train = preprocessor.fit(X_source_train, y_source_train).transform(
            X_source_train
        )

        csp, lda = train_csp_lda(X_source_train, y_source_train, csp_kwargs, lda_kwargs)

        for target in subjects:
            tgt_idx = subject_to_idx[target]
            X_test, y_test = loader.load_subject(target, split="test")
            X_test = preprocessor.transform(X_test)

            if method in ("ea", "ra") and target != source:
                assert domain_mean is not None
                A = compute_alignment_matrix(
                    domain_mean[source], domain_mean[target], eps=cov_eps
                )
                X_eval = apply_alignment(X_test, A)
            else:
                X_eval = X_test

            y_pred = lda.predict(csp.transform(X_eval))
            m = compute_metrics(y_test, y_pred)
            acc[src_idx, tgt_idx] = m["accuracy"]
            kappa[src_idx, tgt_idx] = m["kappa"]
            f1[src_idx, tgt_idx] = m["f1_macro"]

    return {"accuracy": acc, "kappa": kappa, "f1_macro": f1}

