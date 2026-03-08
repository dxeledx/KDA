#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


logger = get_logger("dca_bgf_mvp")


def _parse_targets(value: str) -> List[int]:
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    return items


def _save_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in ["accuracy", "kappa", "f1_macro"]
        if col in df.columns
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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


def _run_loso_target(
    loader: BCIDataLoader,
    subjects: List[int],
    target: int,
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    cov_eps: float,
    model_cfg: Dict,
    out_dir: Path,
    fig_dir: Path,
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
    y_pred, details = dca.predict_online(X_test, y_true=y_test, return_details=True)

    metrics = compute_metrics(y_test, y_pred)

    np.savez(
        out_dir / f"details_targetA{target:02d}.npz",
        y_true=y_test,
        y_pred=y_pred,
        **details,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(details["w"], linewidth=1.5)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Trial")
    plt.ylabel("Alignment weight w")
    plt.title(f"DCA-BGF weight evolution (Target A{target:02d})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"w_evolution_targetA{target:02d}.pdf", dpi=300)
    plt.close()

    row = {"target_subject": target, **metrics}
    logger.info("Target=%s metrics=%s", target, metrics)
    return row


def _run_within_subject(
    loader: BCIDataLoader,
    subject: int,
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    cov_eps: float,
    model_cfg: Dict,
) -> Dict:
    X_train, y_train, X_test, y_test = loader.get_train_test_split(subject)
    X_train = pre.fit(X_train, y_train).transform(X_train)
    X_test = pre.transform(X_test)

    # For sanity check: use train as calibration set (A should be close to identity)
    alignment_matrix = _compute_ra_matrix(X_train, X_train, eps=cov_eps)
    source_covariances = compute_covariances(X_train, eps=cov_eps)

    csp = CSP(**csp_kwargs)
    feats_train = csp.fit_transform(X_train, y_train)
    lda = LDA(**lda_kwargs).fit(feats_train, y_train)

    dca = _build_dca_bgf(
        csp, lda, alignment_matrix, feats_train, source_covariances, model_cfg
    )
    y_pred = dca.predict_online(X_test, return_details=False)
    metrics = compute_metrics(y_test, y_pred)
    return {"subject": subject, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["loso", "within"], default="loso")
    parser.add_argument("--targets", default="1,5,9", help="Comma separated target ids")
    parser.add_argument("--within-subject", type=int, default=1)
    args = parser.parse_args()

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

    base_dir = ensure_dir("results/stage2/mvp")
    fig_dir = ensure_dir("results/stage2/figures")

    if args.mode == "within":
        row = _run_within_subject(
            loader,
            subject=int(args.within_subject),
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            cov_eps=cov_eps,
            model_cfg=model_cfg,
        )
        df = pd.DataFrame([row])
        df.to_csv(base_dir / "within_subject.csv", index=False)
        _save_summary(df, base_dir / "summary_within.json")
        logger.info("Within-subject done. Saved to %s", base_dir)
        return

    targets = _parse_targets(args.targets)
    rows = []
    for target in targets:
        rows.append(
            _run_loso_target(
                loader,
                subjects,
                target=target,
                pre=pre,
                csp_kwargs=csp_kwargs,
                lda_kwargs=lda_kwargs,
                cov_eps=cov_eps,
                model_cfg=model_cfg,
                out_dir=base_dir,
                fig_dir=fig_dir,
            )
        )

    df = pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)
    df.to_csv(base_dir / "loso.csv", index=False)
    _save_summary(df, base_dir / "summary.json")
    logger.info("MVP LOSO done. Saved to %s", base_dir)


if __name__ == "__main__":
    main()
