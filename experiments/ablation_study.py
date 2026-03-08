#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.alignment.behavior_feedback import BehaviorGuidedFeedback
from src.alignment.conditional import FixedWeight, LinearConditionalWeight
from src.alignment.dca_bgf import DCABGF
from src.alignment.euclidean import compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.context import ContextComputer
from src.utils.logger import get_logger


logger = get_logger("ablation_study")


def _save_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {
        "methods": sorted(df["method"].unique().tolist()),
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std()),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _compute_ra_matrix(X_source: np.ndarray, X_target: np.ndarray, eps: float) -> np.ndarray:
    C_source = mean_riemann(compute_covariances(X_source, eps=eps))
    C_target = mean_riemann(compute_covariances(X_target, eps=eps))
    return compute_alignment_matrix(C_source, C_target, eps=eps)


def _make_context_computer(
    source_features: np.ndarray,
    source_covariances: np.ndarray,
    model_cfg: Dict,
) -> ContextComputer:
    dca_cfg = model_cfg.get("dca_bgf", {})
    ctx_dim = int(dca_cfg.get("context_dim", 3))
    ctx_cfg = dca_cfg.get("context", {})
    normalize = bool(ctx_cfg.get("normalize", True))
    feature_names = ctx_cfg.get("features")

    source_mean = np.mean(source_features, axis=0)
    ctx = ContextComputer(
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
        covariance_stream = source_covariances if ctx.requires_trial_covariance else None
        ctx.fit_normalizer(source_features, covariance_stream=covariance_stream)
    return ctx


def _make_feedback(model_cfg: Dict, enabled: bool) -> Tuple[BehaviorGuidedFeedback, bool]:
    dca_cfg = model_cfg.get("dca_bgf", {})
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
    return feedback, bool(enabled)


def _make_linear_conditional(model_cfg: Dict) -> LinearConditionalWeight:
    dca_cfg = model_cfg.get("dca_bgf", {})
    ctx_cfg = dca_cfg.get("context", {})
    feature_names = ctx_cfg.get("features")
    context_dim = len(feature_names) if feature_names is not None else int(dca_cfg.get("context_dim", 3))
    cond_cfg = dca_cfg.get("conditional", {})
    weights = list(cond_cfg.get("weights", [1.2, 0.6, 0.3]))
    if len(weights) < context_dim:
        weights = weights + [weights[0]] * (context_dim - len(weights))
    elif len(weights) > context_dim:
        weights = weights[:context_dim]
    return LinearConditionalWeight(
        weights=weights,
        bias=float(cond_cfg.get("bias", 0.0)),
        temperature=float(cond_cfg.get("temperature", 1.0)),
        ema_smooth_alpha=float(cond_cfg.get("ema_smooth_alpha", 0.2)),
    )


def _run_method(
    csp: CSP,
    lda: LDA,
    alignment_matrix: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    source_features: np.ndarray,
    source_covariances: np.ndarray,
    model_cfg: Dict,
    conditional_weight,
    use_feedback: bool,
) -> Dict:
    context = _make_context_computer(source_features, source_covariances, model_cfg)
    feedback, feedback_enabled = _make_feedback(model_cfg, enabled=use_feedback)

    dca = DCABGF(
        csp=csp,
        lda=lda,
        alignment_matrix=alignment_matrix,
        context_computer=context,
        conditional_weight=conditional_weight,
        behavior_feedback=feedback,
        use_feedback=feedback_enabled,
    )
    y_pred, details = dca.predict_online(X_test, y_true=y_test, return_details=True)
    metrics = compute_metrics(y_test, y_pred)

    w = np.asarray(details["w"], dtype=np.float64)
    w_mean = float(w.mean()) if w.size else 0.0
    w_std = float(w.std()) if w.size else 0.0
    return {**metrics, "w_mean": w_mean, "w_std": w_std}


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

    out_dir = ensure_dir("results/stage2/ablation")
    fig_dir = ensure_dir("results/stage2/figures")

    methods = [
        ("w=0.0", FixedWeight(0.0), False),
        ("w=0.5", FixedWeight(0.5), False),
        ("w=1.0", FixedWeight(1.0), False),
        ("conditional_only", _make_linear_conditional(model_cfg), False),
        ("feedback_only(w=0.5)", FixedWeight(0.5), True),
        ("dca_bgf(full)", _make_linear_conditional(model_cfg), True),
    ]

    rows = []
    for target in subjects:
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

        for method_name, conditional, use_feedback in methods:
            # Ensure per-run internal state is clean
            if hasattr(conditional, "reset"):
                conditional.reset()

            m = _run_method(
                csp=csp,
                lda=lda,
                alignment_matrix=alignment_matrix,
                X_test=X_test,
                y_test=y_test,
                source_features=feats_source,
                source_covariances=source_covariances,
                model_cfg=model_cfg,
                conditional_weight=conditional,
                use_feedback=use_feedback,
            )
            rows.append({"target_subject": target, "method": method_name, **m})
            logger.info("Target=%s method=%s acc=%.4f", target, method_name, m["accuracy"])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_loso.csv", index=False)
    _save_summary(df, out_dir / "summary.json")

    plt.figure(figsize=(11, 4))
    order = [m[0] for m in methods]
    sns.boxplot(data=df, x="method", y="accuracy", order=order, color="steelblue")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_boxplot.pdf", dpi=300)
    plt.close()

    logger.info("Done. Saved to %s", out_dir)


if __name__ == "__main__":
    main()
