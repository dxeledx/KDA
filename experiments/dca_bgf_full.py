#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.behavior_feedback import BehaviorGuidedFeedback
from src.alignment.conditional import FixedWeight, LinearConditionalWeight
from src.alignment.dca_bgf import DCABGF
from src.alignment.euclidean import compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.evaluation.protocols import evaluate_loso
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.context import ContextComputer
from src.utils.logger import get_logger


logger = get_logger("dca_bgf_full")


def _save_summary(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _compute_ra_matrix(X_source: np.ndarray, X_target: np.ndarray, eps: float) -> np.ndarray:
    C_source = mean_riemann(compute_covariances(X_source, eps=eps))
    C_target = mean_riemann(compute_covariances(X_target, eps=eps))
    return compute_alignment_matrix(C_source, C_target, eps=eps)


def _build_dca_bgf(
    csp: CSP,
    lda: LDA,
    alignment_matrix: np.ndarray,
    source_features: np.ndarray,
    source_covariances: np.ndarray,
    model_cfg: Dict,
) -> DCABGF:
    dca_cfg = model_cfg.get("dca_bgf", {})

    ctx_dim = int(dca_cfg.get("context_dim", 3))
    ctx_cfg = dca_cfg.get("context", {})
    normalize = bool(ctx_cfg.get("normalize", True))
    feature_names = ctx_cfg.get("features")
    source_mean = np.mean(source_features, axis=0)
    context_computer = ContextComputer(
        source_mean=source_mean,
        context_dim=ctx_dim,
        feature_names=feature_names,
        ema_alpha=float(ctx_cfg.get("ema_alpha", 0.1)),
        recent_window=int(ctx_cfg.get("recent_window", 5)),
        normalize=normalize,
        source_covariance=mean_riemann(source_covariances),
        cov_eps=float(ctx_cfg.get("cov_eps", 1.0e-6)),
    )
    if normalize:
        covariance_stream = source_covariances if context_computer.requires_trial_covariance else None
        context_computer.fit_normalizer(source_features, covariance_stream=covariance_stream)

    cond_cfg = dca_cfg.get("conditional", {})
    cond_type = str(cond_cfg.get("type", "linear")).lower()
    if cond_type == "linear":
        weights = list(cond_cfg.get("weights", [1.2, 0.6, 0.3]))
        if len(weights) < context_computer.context_dim:
            weights = weights + [weights[0]] * (context_computer.context_dim - len(weights))
        elif len(weights) > context_computer.context_dim:
            weights = weights[: context_computer.context_dim]
        conditional = LinearConditionalWeight(
            weights=weights,
            bias=float(cond_cfg.get("bias", 0.0)),
            temperature=float(cond_cfg.get("temperature", 1.0)),
            ema_smooth_alpha=float(cond_cfg.get("ema_smooth_alpha", 0.2)),
        )
    elif cond_type == "fixed":
        conditional = FixedWeight(float(cond_cfg.get("weight", 0.5)))
    else:
        raise ValueError(f"Unsupported conditional.type: {cond_type}")

    fb_cfg = dca_cfg.get("feedback", {})
    feedback = BehaviorGuidedFeedback(
        window_size=int(fb_cfg.get("window_size", 10)),
        entropy_high_factor=float(fb_cfg.get("entropy_high_factor", 0.8)),
        entropy_low_factor=float(fb_cfg.get("entropy_low_factor", 0.3)),
        conf_trend_threshold=float(fb_cfg.get("conf_trend_threshold", -0.05)),
        alpha=float(fb_cfg.get("alpha", 0.1)),
        beta=float(fb_cfg.get("beta", 0.05)),
        conf_trend_alpha=float(fb_cfg.get("conf_trend_alpha", 0.15)),
        momentum=float(fb_cfg.get("momentum", 0.7)),
        delta_w_max=float(fb_cfg.get("delta_w_max", 0.2)),
        update_every=int(fb_cfg.get("update_every", 1)),
        conflict_mode=str(fb_cfg.get("conflict_mode", "sum")),
    )
    use_feedback = bool(fb_cfg.get("enabled", True))

    return DCABGF(
        csp=csp,
        lda=lda,
        alignment_matrix=alignment_matrix,
        context_computer=context_computer,
        conditional_weight=conditional,
        behavior_feedback=feedback,
        use_feedback=use_feedback,
    )


def _run_target(
    loader: BCIDataLoader,
    subjects: List[int],
    target: int,
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    cov_eps: float,
    model_cfg: Dict,
) -> Dict:
    X_train_list, y_train_list = [], []
    for sid in subjects:
        if sid == target:
            continue
        X_s, y_s = loader.load_subject(sid, split="train")
        X_train_list.append(X_s)
        y_train_list.append(y_s)

    X_source = np.concatenate(X_train_list, axis=0)
    y_source = np.concatenate(y_train_list, axis=0)

    X_t_train, _y_t_train = loader.load_subject(target, split="train")
    X_test, y_test = loader.load_subject(target, split="test")

    X_source = pre.fit(X_source, y_source).transform(X_source)
    X_t_train = pre.transform(X_t_train)
    X_test = pre.transform(X_test)

    alignment_matrix = _compute_ra_matrix(X_source, X_t_train, eps=cov_eps)
    source_covariances = compute_covariances(X_source, eps=cov_eps)

    csp = CSP(**csp_kwargs)
    feats_source = csp.fit_transform(X_source, y_source)
    lda = LDA(**lda_kwargs).fit(feats_source, y_source)

    dca = _build_dca_bgf(
        csp, lda, alignment_matrix, feats_source, source_covariances, model_cfg
    )
    y_pred = dca.predict_online(X_test, return_details=False)

    metrics = compute_metrics(y_test, y_pred)
    row = {"target_subject": target, **metrics}
    logger.info("Target=%s metrics=%s", target, metrics)
    return row


def main() -> None:
    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))

    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    subjects = list(map(int, data_cfg["dataset"]["subjects"]))

    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))

    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    out_dir = ensure_dir("results/stage2/full")

    logger.info("Running RA baseline (LOSO)...")
    ra_df = evaluate_loso(
        loader, subjects, pre, csp_kwargs, lda_kwargs, method="ra", cov_eps=cov_eps
    )
    ra_df.to_csv(out_dir / "ra_loso.csv", index=False)

    logger.info("Running DCA-BGF (LOSO)...")
    rows = []
    for target in subjects:
        rows.append(
            _run_target(
                loader,
                subjects,
                target=target,
                pre=pre,
                csp_kwargs=csp_kwargs,
                lda_kwargs=lda_kwargs,
                cov_eps=cov_eps,
                model_cfg=model_cfg,
            )
        )
    dca_df = pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)
    dca_df.to_csv(out_dir / "dca_bgf_loso.csv", index=False)

    summary = {
        "ra": {
            "accuracy": {"mean": float(ra_df["accuracy"].mean()), "std": float(ra_df["accuracy"].std())},
            "kappa": {"mean": float(ra_df["kappa"].mean()), "std": float(ra_df["kappa"].std())},
            "f1_macro": {"mean": float(ra_df["f1_macro"].mean()), "std": float(ra_df["f1_macro"].std())},
        },
        "dca_bgf": {
            "accuracy": {"mean": float(dca_df["accuracy"].mean()), "std": float(dca_df["accuracy"].std())},
            "kappa": {"mean": float(dca_df["kappa"].mean()), "std": float(dca_df["kappa"].std())},
            "f1_macro": {"mean": float(dca_df["f1_macro"].mean()), "std": float(dca_df["f1_macro"].std())},
        },
    }
    summary["improvement_vs_ra"] = {
        "accuracy": float(summary["dca_bgf"]["accuracy"]["mean"] - summary["ra"]["accuracy"]["mean"]),
        "kappa": float(summary["dca_bgf"]["kappa"]["mean"] - summary["ra"]["kappa"]["mean"]),
        "f1_macro": float(summary["dca_bgf"]["f1_macro"]["mean"] - summary["ra"]["f1_macro"]["mean"]),
    }
    _save_summary(summary, out_dir / "summary.json")

    logger.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
