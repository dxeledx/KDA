#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.conditional import FixedWeight, LinearConditionalWeight
from src.alignment.euclidean import compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("trial_dynamic_gate_exp_a")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_ra_matrix(X_source: np.ndarray, X_target: np.ndarray, eps: float) -> np.ndarray:
    C_source = mean_riemann(compute_covariances(X_source, eps=eps))
    C_target = mean_riemann(compute_covariances(X_target, eps=eps))
    return compute_alignment_matrix(C_source, C_target, eps=eps)


def _build_trial_gate(model_cfg: Dict) -> LinearConditionalWeight:
    dca_cfg = model_cfg.get("dca_bgf", {})
    cond_cfg = dca_cfg.get("conditional", {})
    return LinearConditionalWeight(
        weights=[float(cond_cfg.get("weights", [1.2])[0])],
        bias=float(cond_cfg.get("bias", 0.0)),
        temperature=float(cond_cfg.get("temperature", 1.0)),
        ema_smooth_alpha=float(cond_cfg.get("ema_smooth_alpha", 0.0)),
    )


def _predict_fixed_covariance_space(
    csp: CSP,
    lda: LDA,
    cov_raw: np.ndarray,
    cov_ra: np.ndarray,
    weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cov_mix = (1.0 - float(weight)) * cov_raw + float(weight) * cov_ra
    features = csp.transform_covariances(cov_mix)
    y_pred = lda.predict(features)
    return y_pred.astype(np.int64), features


def _predict_dynamic_covariance_space(
    csp: CSP,
    lda: LDA,
    cov_raw: np.ndarray,
    cov_ra: np.ndarray,
    gate: LinearConditionalWeight,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n_trials = cov_raw.shape[0]
    target_mean = None
    w_values, d_tgt_values = [], []
    features = []

    for trial_idx in range(n_trials):
        current_cov = cov_raw[trial_idx]
        if target_mean is None:
            d_tgt = 0.0
            target_mean = current_cov.copy()
        else:
            d_tgt = float(np.linalg.norm(current_cov - target_mean))
            target_mean = 0.9 * target_mean + 0.1 * current_cov

        w_t = float(gate.predict(np.array([d_tgt], dtype=np.float64)))
        cov_mix = (1.0 - w_t) * current_cov + w_t * cov_ra[trial_idx]
        features.append(csp.transform_covariances(cov_mix[None, ...])[0])
        w_values.append(w_t)
        d_tgt_values.append(d_tgt)

    feature_arr = np.asarray(features, dtype=np.float64)
    y_pred = lda.predict(feature_arr).astype(np.int64)
    details = {
        "w": np.asarray(w_values, dtype=np.float64),
        "d_tgt": np.asarray(d_tgt_values, dtype=np.float64),
        "features_final": feature_arr,
        "trial_index": np.arange(n_trials, dtype=np.int64),
    }
    return y_pred, details


def _plot_subject_details(
    target_subject: int,
    details: Dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    trial_index = details["trial_index"]
    w = details["w"]
    d_tgt = details["d_tgt"]

    plt.figure(figsize=(12, 4))
    plt.plot(trial_index, w, linewidth=1.6)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Trial")
    plt.ylabel("w_t")
    plt.title(f"Exp-A dynamic w_t (Target A{target_subject:02d})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"w_vs_trial_targetA{target_subject:02d}.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(trial_index, d_tgt, linewidth=1.6, color="tab:orange")
    plt.xlabel("Trial")
    plt.ylabel("d_tgt")
    plt.title(f"Exp-A d_tgt (Target A{target_subject:02d})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"d_tgt_vs_trial_targetA{target_subject:02d}.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(d_tgt, w, s=18, alpha=0.75)
    plt.xlabel("d_tgt")
    plt.ylabel("w_t")
    plt.title(f"Exp-A w_t vs d_tgt (Target A{target_subject:02d})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"w_vs_d_tgt_scatter_targetA{target_subject:02d}.pdf", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--targets", default=None, help="Comma separated target subjects")
    args = parser.parse_args()

    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))
    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    all_subjects = list(map(int, data_cfg["dataset"]["subjects"]))
    if args.targets:
        wanted = {int(token.strip()) for token in args.targets.split(",") if token.strip()}
        target_subjects = [subject for subject in all_subjects if subject in wanted]
    else:
        target_subjects = list(all_subjects)

    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))
    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/stage3/trial_dynamic_gate_exp_a/{args.run_name}")
    details_dir = ensure_dir(root_dir / "details")
    fig_dir = ensure_dir(root_dir / "figures")

    dynamic_rows: List[Dict] = []
    comparison_rows: List[Dict] = []

    for target_subject in target_subjects:
        X_source_blocks, y_source_blocks = [], []
        for subject in all_subjects:
            if subject == target_subject:
                continue
            X_train_subject, y_train_subject = loader.load_subject(subject, split="train")
            X_source_blocks.append(X_train_subject)
            y_source_blocks.append(y_train_subject)

        X_source_raw = np.concatenate(X_source_blocks, axis=0)
        y_source = np.concatenate(y_source_blocks, axis=0)
        X_target_train_raw, _ = loader.load_subject(target_subject, split="train")
        X_target_test_raw, y_target_test = loader.load_subject(target_subject, split="test")

        X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
        X_target_train = pre.transform(X_target_train_raw)
        X_target_test = pre.transform(X_target_test_raw)

        cov_raw = compute_covariances(X_target_test, eps=cov_eps)
        alignment_matrix = _compute_ra_matrix(X_source, X_target_train, eps=cov_eps)
        X_target_ra = np.einsum("ij,tjk,lk->til", alignment_matrix, cov_raw, alignment_matrix)

        csp = CSP(**csp_kwargs)
        features_source = csp.fit_transform(X_source, y_source)
        lda = LDA(**lda_kwargs).fit(features_source, y_source)

        gate = _build_trial_gate(model_cfg)
        y_pred_dynamic, details = _predict_dynamic_covariance_space(
            csp=csp,
            lda=lda,
            cov_raw=cov_raw,
            cov_ra=X_target_ra,
            gate=gate,
        )

        fixed_results = {}
        for weight in [1.0, 0.5, 0.0]:
            y_pred_fixed, _features = _predict_fixed_covariance_space(
                csp=csp,
                lda=lda,
                cov_raw=cov_raw,
                cov_ra=X_target_ra,
                weight=weight,
            )
            fixed_results[weight] = y_pred_fixed
            comparison_rows.append(
                {
                    "target_subject": int(target_subject),
                    "method": f"fixed_w={weight:.1f}",
                    **compute_metrics(y_target_test, y_pred_fixed),
                }
            )

        dynamic_metrics = compute_metrics(y_target_test, y_pred_dynamic)
        dynamic_rows.append(
            {
                "target_subject": int(target_subject),
                **dynamic_metrics,
                "w_mean": float(details["w"].mean()),
                "w_std": float(details["w"].std()),
            }
        )
        comparison_rows.append(
            {
                "target_subject": int(target_subject),
                "method": "dynamic",
                **dynamic_metrics,
            }
        )

        np.savez(
            details_dir / f"subject_A{target_subject:02d}.npz",
            y_true=np.asarray(y_target_test, dtype=np.int64),
            y_pred_dynamic=y_pred_dynamic,
            y_pred_w1=fixed_results[1.0],
            y_pred_w05=fixed_results[0.5],
            y_pred_w0=fixed_results[0.0],
            w=details["w"],
            d_tgt=details["d_tgt"],
            trial_index=details["trial_index"],
        )
        _plot_subject_details(target_subject=target_subject, details=details, fig_dir=fig_dir)
        logger.info(
            "Exp-A target=%s dynamic_acc=%.4f w_mean=%.4f",
            target_subject,
            dynamic_metrics["accuracy"],
            float(details["w"].mean()),
        )

    dynamic_df = pd.DataFrame(dynamic_rows).sort_values("target_subject").reset_index(drop=True)
    comparison_df = pd.DataFrame(comparison_rows).sort_values(["target_subject", "method"]).reset_index(drop=True)
    dynamic_df.to_csv(root_dir / "loso_dynamic.csv", index=False)
    comparison_df.to_csv(root_dir / "comparison.csv", index=False)

    summary = {
        "dynamic": {
            "accuracy_mean": float(dynamic_df["accuracy"].mean()),
            "accuracy_std": float(dynamic_df["accuracy"].std(ddof=1)) if len(dynamic_df) > 1 else 0.0,
            "w_mean": float(dynamic_df["w_mean"].mean()),
            "w_std_mean": float(dynamic_df["w_std"].mean()),
        },
        "fixed": {
            method: {
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
            for method, group in comparison_df[comparison_df["method"] != "dynamic"].groupby("method")
        },
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.5, 4.5))
    plot_df = comparison_df.copy()
    order = ["fixed_w=1.0", "fixed_w=0.5", "fixed_w=0.0", "dynamic"]
    data = [plot_df.loc[plot_df["method"] == method, "accuracy"].to_numpy(dtype=np.float64) for method in order]
    plt.boxplot(data, labels=order)
    plt.ylabel("Accuracy")
    plt.title("Exp-A dynamic vs fixed baselines")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "dynamic_vs_fixed_boxplot.pdf", dpi=300)
    plt.close()
    logger.info("Saved Exp-A outputs to %s", root_dir)


if __name__ == "__main__":
    main()
