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

from src.alignment.conditional import LinearConditionalWeight
from src.alignment.euclidean import compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("trial_dynamic_gate_exp_b1")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_ra_matrix(X_source: np.ndarray, X_target: np.ndarray, eps: float) -> np.ndarray:
    C_source = mean_riemann(compute_covariances(X_source, eps=eps))
    C_target = mean_riemann(compute_covariances(X_target, eps=eps))
    return compute_alignment_matrix(C_source, C_target, eps=eps)


def compute_trial_geometry_context(
    covariances: np.ndarray,
    source_mean_covariance: np.ndarray,
    ema_alpha: float = 0.1,
    recent_window: int = 5,
) -> np.ndarray:
    covariances = np.asarray(covariances, dtype=np.float64)
    source_mean_covariance = np.asarray(source_mean_covariance, dtype=np.float64)
    target_mean = None
    history: List[np.ndarray] = []
    contexts: List[np.ndarray] = []

    for covariance in covariances:
        d_src = float(np.linalg.norm(covariance - source_mean_covariance, ord="fro"))
        if target_mean is None:
            d_tgt = 0.0
            target_mean = covariance.copy()
        else:
            d_tgt = float(np.linalg.norm(covariance - target_mean, ord="fro"))
            target_mean = (1.0 - float(ema_alpha)) * target_mean + float(ema_alpha) * covariance

        sigma_recent = 0.0
        if len(history) >= int(recent_window):
            recent = np.stack(history[-int(recent_window) :], axis=0)
            sigma_recent = float(np.std(recent))

        contexts.append(np.asarray([d_src, d_tgt, sigma_recent], dtype=np.float64))
        history.append(covariance.copy())

    return np.asarray(contexts, dtype=np.float64)


def fit_source_context_normalizer(
    source_covariances: np.ndarray,
    source_mean_covariance: np.ndarray,
    ema_alpha: float = 0.1,
    recent_window: int = 5,
) -> Dict[str, np.ndarray]:
    context = compute_trial_geometry_context(
        covariances=source_covariances,
        source_mean_covariance=source_mean_covariance,
        ema_alpha=ema_alpha,
        recent_window=recent_window,
    )
    return {
        "mean": context.mean(axis=0),
        "std": np.clip(context.std(axis=0), 1.0e-8, None),
    }


def normalize_trial_geometry_context(
    context_raw: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    context_raw = np.asarray(context_raw, dtype=np.float64)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    return (context_raw - mean) / std


def _build_trial_gate(model_cfg: Dict) -> LinearConditionalWeight:
    dca_cfg = model_cfg.get("dca_bgf", {})
    cond_cfg = dca_cfg.get("conditional", {})
    base_weights = list(cond_cfg.get("weights", [1.2, 0.6, 0.3]))
    if len(base_weights) < 3:
        base_weights = base_weights + [base_weights[0]] * (3 - len(base_weights))
    weights = [float(value) for value in base_weights[:3]]
    return LinearConditionalWeight(
        weights=weights,
        bias=float(cond_cfg.get("bias", 0.0)),
        temperature=float(cond_cfg.get("temperature", 1.0)),
        ema_smooth_alpha=float(cond_cfg.get("ema_smooth_alpha", 0.2)),
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


def predict_dynamic_covariance_space_normalized(
    csp: CSP,
    lda: LDA,
    cov_raw: np.ndarray,
    cov_ra: np.ndarray,
    gate: LinearConditionalWeight,
    context_raw: np.ndarray,
    context_norm: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cov_raw = np.asarray(cov_raw, dtype=np.float64)
    cov_ra = np.asarray(cov_ra, dtype=np.float64)
    context_raw = np.asarray(context_raw, dtype=np.float64)
    context_norm = np.asarray(context_norm, dtype=np.float64)

    features = []
    w_values = []
    for trial_idx in range(cov_raw.shape[0]):
        w_t = float(gate.predict(context_norm[trial_idx]))
        cov_mix = (1.0 - w_t) * cov_raw[trial_idx] + w_t * cov_ra[trial_idx]
        features.append(csp.transform_covariances(cov_mix[None, ...])[0])
        w_values.append(w_t)

    feature_arr = np.asarray(features, dtype=np.float64)
    y_pred = lda.predict(feature_arr).astype(np.int64)
    details = {
        "w": np.asarray(w_values, dtype=np.float64),
        "context_raw": context_raw.astype(np.float64),
        "context_norm": context_norm.astype(np.float64),
        "d_src": context_raw[:, 0].astype(np.float64),
        "d_tgt": context_raw[:, 1].astype(np.float64),
        "sigma_recent": context_raw[:, 2].astype(np.float64),
        "trial_index": np.arange(cov_raw.shape[0], dtype=np.int64),
    }
    return y_pred, details


def _plot_subject_details(
    target_subject: int,
    details: Dict[str, np.ndarray],
    fig_dir: Path,
) -> None:
    trial_index = details["trial_index"]
    w = details["w"]
    context_raw = details["context_raw"]
    context_norm = details["context_norm"]
    d_src = details["d_src"]
    d_tgt = details["d_tgt"]
    sigma_recent = details["sigma_recent"]

    plt.figure(figsize=(12, 4))
    plt.plot(trial_index, w, linewidth=1.6)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Trial")
    plt.ylabel("w_t")
    plt.title(f"Exp-B.1 dynamic w_t (Target A{target_subject:02d})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"w_vs_trial_targetA{target_subject:02d}.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(trial_index, context_raw[:, 0], label="d_src", linewidth=1.5)
    plt.plot(trial_index, context_raw[:, 1], label="d_tgt", linewidth=1.5)
    plt.plot(trial_index, context_raw[:, 2], label="sigma_recent", linewidth=1.5)
    plt.xlabel("Trial")
    plt.ylabel("Raw geometry")
    plt.title(f"Exp-B.1 raw geometry (Target A{target_subject:02d})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"geometry_raw_vs_trial_targetA{target_subject:02d}.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(trial_index, context_norm[:, 0], label="d_src_z", linewidth=1.5)
    plt.plot(trial_index, context_norm[:, 1], label="d_tgt_z", linewidth=1.5)
    plt.plot(trial_index, context_norm[:, 2], label="sigma_recent_z", linewidth=1.5)
    plt.xlabel("Trial")
    plt.ylabel("Normalized geometry")
    plt.title(f"Exp-B.1 normalized geometry (Target A{target_subject:02d})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"geometry_norm_vs_trial_targetA{target_subject:02d}.pdf", dpi=300)
    plt.close()

    for name, values in [
        ("d_tgt", d_tgt),
        ("d_src", d_src),
        ("sigma_recent", sigma_recent),
    ]:
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(values, w, s=18, alpha=0.75)
        plt.xlabel(name)
        plt.ylabel("w_t")
        plt.title(f"Exp-B.1 w_t vs {name} (Target A{target_subject:02d})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"w_vs_{name}_scatter_targetA{target_subject:02d}.pdf", dpi=300)
        plt.close()


def _load_reference_loso(reference_dir: Path) -> pd.DataFrame:
    path = reference_dir / "loso_dynamic.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference file: {path}")
    return pd.read_csv(path).sort_values("target_subject").reset_index(drop=True)


def _build_delta_table(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    merged = current_df.merge(
        reference_df[["target_subject", "accuracy", "kappa", "f1_macro", "w_std"]],
        on="target_subject",
        suffixes=("_current", suffix),
        how="inner",
    )
    merged[f"accuracy_delta{suffix}"] = merged["accuracy_current"] - merged[f"accuracy{suffix}"]
    merged[f"kappa_delta{suffix}"] = merged["kappa_current"] - merged[f"kappa{suffix}"]
    merged[f"f1_macro_delta{suffix}"] = merged["f1_macro_current"] - merged[f"f1_macro{suffix}"]
    merged[f"w_std_delta{suffix}"] = merged["w_std_current"] - merged[f"w_std{suffix}"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--targets", default=None, help="Comma separated target subjects")
    parser.add_argument(
        "--exp-a-dir",
        default="results/stage3/trial_dynamic_gate_exp_a/2026-03-07-trial-dynamic-exp-a",
    )
    parser.add_argument(
        "--exp-b-dir",
        default="results/stage3/trial_dynamic_gate_exp_b/2026-03-07-trial-dynamic-exp-b",
    )
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
    exp_a_reference = _load_reference_loso(Path(args.exp_a_dir))
    exp_b_reference = _load_reference_loso(Path(args.exp_b_dir))

    root_dir = ensure_dir(f"results/stage3/trial_dynamic_gate_exp_b1/{args.run_name}")
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

        cov_source = compute_covariances(X_source, eps=cov_eps)
        cov_raw = compute_covariances(X_target_test, eps=cov_eps)
        alignment_matrix = _compute_ra_matrix(X_source, X_target_train, eps=cov_eps)
        cov_ra = np.einsum("ij,tjk,lk->til", alignment_matrix, cov_raw, alignment_matrix)
        source_mean_cov = mean_riemann(cov_source)

        csp = CSP(**csp_kwargs)
        features_source = csp.fit_transform(X_source, y_source)
        lda = LDA(**lda_kwargs).fit(features_source, y_source)

        context_raw = compute_trial_geometry_context(
            covariances=cov_raw,
            source_mean_covariance=source_mean_cov,
            ema_alpha=0.1,
            recent_window=5,
        )
        stats = fit_source_context_normalizer(
            source_covariances=cov_source,
            source_mean_covariance=source_mean_cov,
            ema_alpha=0.1,
            recent_window=5,
        )
        context_norm = normalize_trial_geometry_context(context_raw, stats)

        gate = _build_trial_gate(model_cfg=model_cfg)
        y_pred_dynamic, details = predict_dynamic_covariance_space_normalized(
            csp=csp,
            lda=lda,
            cov_raw=cov_raw,
            cov_ra=cov_ra,
            gate=gate,
            context_raw=context_raw,
            context_norm=context_norm,
        )

        fixed_results = {}
        for weight in [1.0, 0.5, 0.0]:
            y_pred_fixed, _features = _predict_fixed_covariance_space(
                csp=csp,
                lda=lda,
                cov_raw=cov_raw,
                cov_ra=cov_ra,
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
                "method": "dynamic_exp_b1",
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
            context_raw=details["context_raw"],
            context_norm=details["context_norm"],
            d_src=details["d_src"],
            d_tgt=details["d_tgt"],
            sigma_recent=details["sigma_recent"],
            trial_index=details["trial_index"],
        )
        _plot_subject_details(target_subject=target_subject, details=details, fig_dir=fig_dir)
        logger.info(
            "Exp-B.1 target=%s dynamic_acc=%.4f w_mean=%.4f w_std=%.4f",
            target_subject,
            dynamic_metrics["accuracy"],
            float(details["w"].mean()),
            float(details["w"].std()),
        )

    dynamic_df = pd.DataFrame(dynamic_rows).sort_values("target_subject").reset_index(drop=True)
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["target_subject", "method"]
    ).reset_index(drop=True)
    dynamic_df.to_csv(root_dir / "loso_dynamic.csv", index=False)
    comparison_df.to_csv(root_dir / "comparison.csv", index=False)

    expb1_vs_expa = _build_delta_table(dynamic_df, exp_a_reference, "_exp_a")
    expb1_vs_expb = _build_delta_table(dynamic_df, exp_b_reference, "_exp_b")
    expb1_vs_expa.to_csv(root_dir / "expb1_vs_expa.csv", index=False)
    expb1_vs_expb.to_csv(root_dir / "expb1_vs_expb.csv", index=False)

    summary = {
        "dynamic_exp_b1": {
            "accuracy_mean": float(dynamic_df["accuracy"].mean()),
            "accuracy_std": float(dynamic_df["accuracy"].std(ddof=1))
            if len(dynamic_df) > 1
            else 0.0,
            "w_mean": float(dynamic_df["w_mean"].mean()),
            "w_std_mean": float(dynamic_df["w_std"].mean()),
        },
        "fixed": {
            method: {
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_std": float(group["accuracy"].std(ddof=1))
                if len(group) > 1
                else 0.0,
            }
            for method, group in comparison_df[comparison_df["method"] != "dynamic_exp_b1"].groupby(
                "method"
            )
        },
        "vs_exp_a": {
            "accuracy_mean_delta": float(expb1_vs_expa["accuracy_delta_exp_a"].mean()),
            "kappa_mean_delta": float(expb1_vs_expa["kappa_delta_exp_a"].mean()),
            "f1_macro_mean_delta": float(expb1_vs_expa["f1_macro_delta_exp_a"].mean()),
            "w_std_mean_delta": float(expb1_vs_expa["w_std_delta_exp_a"].mean()),
            "wins_vs_exp_a": int(np.sum(expb1_vs_expa["accuracy_delta_exp_a"] > 0.0)),
            "losses_vs_exp_a": int(np.sum(expb1_vs_expa["accuracy_delta_exp_a"] < 0.0)),
        },
        "vs_exp_b": {
            "accuracy_mean_delta": float(expb1_vs_expb["accuracy_delta_exp_b"].mean()),
            "kappa_mean_delta": float(expb1_vs_expb["kappa_delta_exp_b"].mean()),
            "f1_macro_mean_delta": float(expb1_vs_expb["f1_macro_delta_exp_b"].mean()),
            "w_std_mean_delta": float(expb1_vs_expb["w_std_delta_exp_b"].mean()),
            "wins_vs_exp_b": int(np.sum(expb1_vs_expb["accuracy_delta_exp_b"] > 0.0)),
            "losses_vs_exp_b": int(np.sum(expb1_vs_expb["accuracy_delta_exp_b"] < 0.0)),
        },
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.5, 4.5))
    plot_df = comparison_df.copy()
    order = ["fixed_w=1.0", "fixed_w=0.5", "fixed_w=0.0", "dynamic_exp_b1"]
    data = [
        plot_df.loc[plot_df["method"] == method, "accuracy"].to_numpy(dtype=np.float64)
        for method in order
    ]
    plt.boxplot(data, labels=order)
    plt.ylabel("Accuracy")
    plt.title("Exp-B.1 dynamic vs fixed baselines")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "dynamic_vs_fixed_boxplot.pdf", dpi=300)
    plt.close()
    logger.info("Saved Exp-B.1 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
