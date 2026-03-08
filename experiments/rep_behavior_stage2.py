#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.behavior_feedback import BehaviorGuidedFeedback
from src.alignment.conditional import FixedWeight, LinearConditionalWeight
from src.alignment.dca_bgf import DCABGF
from src.alignment.euclidean import compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import cka
from src.evaluation.visualization import plot_correlation_comparison, plot_scatter
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.context import ContextComputer
from src.utils.logger import get_logger


logger = get_logger("rep_behavior_stage2")


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


def _load_baseline_r(path: Path) -> float:
    if not path.exists():
        return float("nan")
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj.get("r", float("nan")))


def main() -> None:
    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))

    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    subjects = list(map(int, data_cfg["dataset"]["subjects"]))
    n = len(subjects)

    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))

    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    out_dir = ensure_dir("results/stage2/rep_behavior")
    fig_dir = ensure_dir("results/stage2/figures")

    acc_mat = np.zeros((n, n), dtype=np.float64)
    rep_sim = np.eye(n, dtype=np.float64)

    subject_to_idx = {sid: i for i, sid in enumerate(subjects)}

    pair_rows: List[Dict] = []

    for source in subjects:
        src_idx = subject_to_idx[source]

        X_src_train, y_src_train = loader.load_subject(source, split="train")
        X_src_test, y_src_test = loader.load_subject(source, split="test")
        X_src_train = pre.fit(X_src_train, y_src_train).transform(X_src_train)
        X_src_test = pre.transform(X_src_test)

        csp = CSP(**csp_kwargs)
        feats_src_train = csp.fit_transform(X_src_train, y_src_train)
        lda = LDA(**lda_kwargs).fit(feats_src_train, y_src_train)

        F_src_test = csp.transform(X_src_test)

        for target in subjects:
            tgt_idx = subject_to_idx[target]

            X_t_train, _y_t_train = loader.load_subject(target, split="train")
            X_t_test, y_t_test = loader.load_subject(target, split="test")
            X_t_train = pre.transform(X_t_train)
            X_t_test = pre.transform(X_t_test)

            alignment_matrix = _compute_ra_matrix(X_src_train, X_t_train, eps=cov_eps)
            source_covariances = compute_covariances(X_src_train, eps=cov_eps)
            dca = _build_dca_bgf(
                csp,
                lda,
                alignment_matrix,
                feats_src_train,
                source_covariances,
                model_cfg,
            )
            y_pred, details = dca.predict_online(
                X_t_test, y_true=y_t_test, return_details=True, return_features=True
            )

            acc = float((y_pred == y_t_test).mean())
            acc_mat[src_idx, tgt_idx] = acc

            F_tgt_dyn = details["features_final"]
            min_n = int(min(F_src_test.shape[0], F_tgt_dyn.shape[0]))
            rep_val = cka(F_src_test[:min_n], F_tgt_dyn[:min_n])
            rep_sim[src_idx, tgt_idx] = rep_val

            pair_rows.append(
                {
                    "source": source,
                    "target": target,
                    "accuracy": acc,
                    "cka": float(rep_val),
                }
            )

            logger.info(
                "Pair A%02d->A%02d acc=%.4f cka=%.4f",
                source,
                target,
                acc,
                rep_val,
            )

    np.save(out_dir / "transfer_accuracy.npy", acc_mat)
    np.save(out_dir / "rep_sim.npy", rep_sim)
    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_metrics.csv", index=False)

    rep_values = rep_sim[~np.eye(n, dtype=bool)]
    beh_values = acc_mat[~np.eye(n, dtype=bool)]
    r, p = pearsonr(rep_values, beh_values)

    (out_dir / "correlation.json").write_text(
        json.dumps({"r": float(r), "p_value": float(p)}, indent=2), encoding="utf-8"
    )

    plot_scatter(
        rep_values,
        beh_values,
        save_path=fig_dir / "rep_acc_scatter_dca_bgf.pdf",
        title="Representation-Behavior Consistency (DCA-BGF)",
        r=float(r),
        p_value=float(p),
    )

    # Compare with stage-1 baselines if available
    baseline_methods = [("No Alignment", "noalign"), ("EA", "ea"), ("RA", "ra")]
    correlations: List[float] = []
    method_names: List[str] = []
    for name, key in baseline_methods:
        r0 = _load_baseline_r(Path(f"results/baselines/{key}/correlation.json"))
        if np.isfinite(r0):
            method_names.append(name)
            correlations.append(float(r0))
    method_names.append("DCA-BGF")
    correlations.append(float(r))

    plot_correlation_comparison(
        method_names,
        correlations,
        save_path=fig_dir / "correlation_comparison_stage2.pdf",
        title="Representation-Behavior Correlation (Stage2)",
    )

    logger.info("DCA-BGF correlation: r=%.3f p=%.3g", float(r), float(p))
    logger.info("Done. Saved to %s", out_dir)


if __name__ == "__main__":
    main()
