#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.euclidean import apply_alignment, compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.kcar_analysis import (
    compare_window_scores,
    compute_kcar,
    compute_transition_residuals,
    fit_koopman_operator,
    fit_subjectwise_global_koopman,
    fit_tangent_projector,
    label_window_alignment_risk,
)
from src.evaluation.kcar_policy import add_budget_rank_columns, make_near_causal_scores
from src.features.covariance import compute_covariances
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.context import ContextComputer
from src.utils.logger import get_logger
from src.utils.monitoring import confidence, entropy


logger = get_logger("kcar_risk_diagnostic")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_targets(all_subjects: Sequence[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def _window_slices(n_trials: int, window_size: int) -> List[Tuple[int, int]]:
    slices = []
    for start in range(0, n_trials, window_size):
        end = start + window_size
        if end <= n_trials:
            slices.append((start, end))
    return slices


def _compute_ra_matrix(X_source: np.ndarray, X_target: np.ndarray, eps: float) -> np.ndarray:
    source_mean = mean_riemann(compute_covariances(X_source, eps=eps))
    target_mean = mean_riemann(compute_covariances(X_target, eps=eps))
    return compute_alignment_matrix(source_mean, target_mean, eps=eps)


def _build_trial_heuristics(
    source_features: np.ndarray,
    features_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = ContextComputer(
        source_mean=np.mean(source_features, axis=0),
        feature_names=["d_src", "d_tgt", "sigma_recent"],
        normalize=False,
    )
    history: List[Dict] = []
    d_src_values, d_tgt_values, sigma_values = [], [], []
    for feature in features_target:
        context = ctx.compute(feature, history)
        d_src_values.append(float(context[0]))
        d_tgt_values.append(float(context[1]))
        sigma_values.append(float(context[2]))
        history.append({"x": np.asarray(feature, dtype=np.float64)})
    return (
        np.asarray(d_src_values, dtype=np.float64),
        np.asarray(d_tgt_values, dtype=np.float64),
        np.asarray(sigma_values, dtype=np.float64),
    )


def _run_subject_diagnostic(
    loader: BCIDataLoader,
    all_subjects: Sequence[int],
    target_subject: int,
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    cov_eps: float,
    window_size: int,
    pca_rank: int,
    ridge_alpha: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    source_trials_raw, source_labels, source_cov_blocks = [], [], []
    for subject in all_subjects:
        if subject == target_subject:
            continue
        X_train_subject, y_train_subject = loader.load_subject(subject, split="train")
        source_trials_raw.append(X_train_subject)
        source_labels.append(y_train_subject)

    X_source_raw = np.concatenate(source_trials_raw, axis=0)
    y_source = np.concatenate(source_labels, axis=0)

    X_target_train_raw, _ = loader.load_subject(target_subject, split="train")
    X_target_test_raw, y_target_test = loader.load_subject(target_subject, split="test")

    X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
    X_target_train = pre.transform(X_target_train_raw)
    X_target_test = pre.transform(X_target_test_raw)

    source_block_lengths = []
    transformed_source_blocks = []
    for X_subject in source_trials_raw:
        X_subject_pre = pre.transform(X_subject)
        transformed_source_blocks.append(X_subject_pre)
        source_block_lengths.append(int(X_subject_pre.shape[0]))
        source_cov_blocks.append(compute_covariances(X_subject_pre, eps=cov_eps))

    source_covariances = np.concatenate(source_cov_blocks, axis=0)
    target_covariances = compute_covariances(X_target_test, eps=cov_eps)
    reference_covariance = mean_riemann(source_covariances)
    tangent_projector = fit_tangent_projector(
        source_covariances=source_covariances,
        reference_covariance=reference_covariance,
        pca_rank=pca_rank,
    )
    source_state_blocks = [tangent_projector.transform(block) for block in source_cov_blocks]
    global_koopman = fit_subjectwise_global_koopman(
        source_state_blocks, ridge_alpha=ridge_alpha
    )
    target_states = tangent_projector.transform(target_covariances)

    alignment_matrix = _compute_ra_matrix(X_source, X_target_train, eps=cov_eps)
    X_target_ra = apply_alignment(X_target_test, alignment_matrix)

    csp = CSP(**csp_kwargs)
    features_source = csp.fit_transform(X_source, y_source)
    lda = LDA(**lda_kwargs).fit(features_source, y_source)

    features_raw = csp.transform(X_target_test)
    features_ra = csp.transform(X_target_ra)
    features_w05 = 0.5 * features_raw + 0.5 * features_ra

    proba_raw = lda.predict_proba(features_raw)
    proba_ra = lda.predict_proba(features_ra)
    proba_w05 = lda.predict_proba(features_w05)

    y_pred_raw = np.argmax(proba_raw, axis=1)
    y_pred_ra = np.argmax(proba_ra, axis=1)
    y_pred_w05 = np.argmax(proba_w05, axis=1)

    correct_raw = (y_pred_raw == y_target_test).astype(np.float64)
    correct_ra = (y_pred_ra == y_target_test).astype(np.float64)
    correct_w05 = (y_pred_w05 == y_target_test).astype(np.float64)

    d_src, d_tgt, sigma_recent = _build_trial_heuristics(features_source, features_raw)
    entropies = np.asarray([entropy(p) for p in proba_ra], dtype=np.float64)
    conf_max = np.asarray([confidence(p) for p in proba_ra], dtype=np.float64)

    rows: List[Dict] = []
    window_rho, window_acc, window_delta, window_labels = [], [], [], []
    window_id_by_trial = np.full(len(y_target_test), -1, dtype=np.int64)
    for window_id, (start, end) in enumerate(_window_slices(len(y_target_test), window_size)):
        states_window = target_states[start:end]
        local_koopman = fit_koopman_operator(states_window, ridge_alpha=ridge_alpha)
        source_residuals = compute_transition_residuals(states_window, global_koopman)
        target_residuals = compute_transition_residuals(states_window, local_koopman)
        rho = compute_kcar(source_residuals, target_residuals)

        acc_ra = float(correct_ra[start:end].mean())
        acc_w0 = float(correct_raw[start:end].mean())
        acc_w05 = float(correct_w05[start:end].mean())
        label_info = label_window_alignment_risk(
            acc_ra=acc_ra,
            acc_w0=acc_w0,
            acc_w05=acc_w05,
            window_size=window_size,
        )

        d_src_window = float(d_src[start:end].mean())
        d_tgt_window = float(d_tgt[start:end].mean())
        sigma_window = float(sigma_recent[start:end].mean())
        entropy_window = float(entropies[start:end].mean())
        conf_window = float(conf_max[start:end].mean())
        rows.append(
            {
                "subject": int(target_subject),
                "window_id": int(window_id),
                "start_trial": int(start),
                "end_trial": int(end),
                "acc_ra": acc_ra,
                "acc_w0": acc_w0,
                "acc_w05": acc_w05,
                "delta_dev_vs_ra": float(label_info["delta_dev_vs_ra"]),
                "rho_kcar": float(rho),
                "rho_kcar_retro": float(rho),
                "d_src": d_src_window,
                "d_src_retro": d_src_window,
                "d_tgt": d_tgt_window,
                "d_tgt_retro": d_tgt_window,
                "sigma_recent": sigma_window,
                "sigma_recent_retro": sigma_window,
                "entropy": entropy_window,
                "entropy_retro": entropy_window,
                "conf_max": conf_window,
                "conf_max_retro": conf_window,
                "label": str(label_info["label"]),
            }
        )

        window_rho.append(float(rho))
        window_acc.append(acc_ra)
        window_delta.append(float(label_info["delta_dev_vs_ra"]))
        window_labels.append(str(label_info["label"]))
        window_id_by_trial[start:end] = int(window_id)

    trajectories = {
        "rho": np.asarray(window_rho, dtype=np.float64),
        "acc_ra": np.asarray(window_acc, dtype=np.float64),
        "delta_dev_vs_ra": np.asarray(window_delta, dtype=np.float64),
        "label": np.asarray(window_labels),
    }
    details = {
        "y_true": np.asarray(y_target_test, dtype=np.int64),
        "y_pred_ra": np.asarray(y_pred_ra, dtype=np.int64),
        "y_pred_w05": np.asarray(y_pred_w05, dtype=np.int64),
        "window_id_by_trial": window_id_by_trial,
    }
    return pd.DataFrame(rows), trajectories, details


def _plot_rho_vs_accuracy(
    trajectories_by_subject: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    subjects = sorted(trajectories_by_subject.keys())
    n_subjects = len(subjects)
    ncols = 3
    nrows = int(np.ceil(n_subjects / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)

    for axis in axes.flat:
        axis.axis("off")

    for axis, subject in zip(axes.flat, subjects):
        axis.axis("on")
        data = trajectories_by_subject[subject]
        x_axis = np.arange(len(data["rho"]))
        axis.plot(x_axis, data["rho"], color="tab:red", linewidth=1.8, label="rho_kcar")
        axis.set_title(f"Target A{subject:02d}")
        axis.set_xlabel("Window")
        axis.set_ylabel("rho_kcar", color="tab:red")
        axis.tick_params(axis="y", labelcolor="tab:red")
        axis.grid(alpha=0.3)
        twin = axis.twinx()
        twin.plot(x_axis, data["acc_ra"], color="tab:blue", linewidth=1.5, label="acc_ra")
        twin.set_ylabel("RA accuracy", color="tab:blue")
        twin.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_rho_vs_delta(window_metrics: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(
        window_metrics["rho_kcar"].to_numpy(dtype=np.float64),
        window_metrics["delta_dev_vs_ra"].to_numpy(dtype=np.float64),
        alpha=0.75,
        s=24,
    )
    plt.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    plt.xlabel("rho_kcar")
    plt.ylabel("delta_dev_vs_ra")
    plt.title("KCAR risk vs deviation gain")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_rho_distribution(window_metrics: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.0, 4.5))
    labels = ["deviation-beneficial", "ra-safe"]
    data = [
        window_metrics.loc[window_metrics["label"] == label, "rho_kcar"].to_numpy(dtype=np.float64)
        for label in labels
    ]
    plt.boxplot(data, labels=labels)
    plt.ylabel("rho_kcar")
    plt.title("KCAR distribution by window label")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--targets", default=None, help="Comma separated target subjects")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--pca-rank", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    args = parser.parse_args()

    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))

    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")
    loader = BCIDataLoader.from_config(data_cfg)
    all_subjects = list(map(int, data_cfg["dataset"]["subjects"]))
    target_subjects = _resolve_targets(all_subjects, args.targets)

    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))
    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/stage3/kcar_diagnostic/{args.run_name}")
    figures_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    subject_frames: List[pd.DataFrame] = []
    trajectories_by_subject: Dict[int, Dict[str, np.ndarray]] = {}
    for target_subject in target_subjects:
        logger.info("Running KCAR diagnostic for target A%02d", target_subject)
        frame, trajectories, details = _run_subject_diagnostic(
            loader=loader,
            all_subjects=all_subjects,
            target_subject=target_subject,
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            cov_eps=cov_eps,
            window_size=int(args.window_size),
            pca_rank=int(args.pca_rank),
            ridge_alpha=float(args.ridge_alpha),
        )
        subject_frames.append(frame)
        trajectories_by_subject[int(target_subject)] = trajectories
        np.savez(details_dir / f"subject_A{int(target_subject):02d}.npz", **details)

    window_metrics = pd.concat(subject_frames, axis=0, ignore_index=True)
    for retro_column in ["rho_kcar_retro", "d_tgt_retro", "sigma_recent_retro"]:
        window_metrics = make_near_causal_scores(window_metrics, score_column=retro_column)
    window_metrics = add_budget_rank_columns(
        window_metrics,
        score_columns=[
            "rho_kcar_retro",
            "rho_kcar_causal",
            "d_tgt_retro",
            "d_tgt_causal",
            "sigma_recent_retro",
            "sigma_recent_causal",
        ],
    )
    window_metrics.to_csv(root_dir / "window_metrics.csv", index=False)

    comparison_df, summary = compare_window_scores(
        window_metrics,
        score_columns=["rho_kcar", "d_src", "d_tgt", "sigma_recent", "entropy", "conf_max"],
    )
    comparison_df.to_csv(root_dir / "comparison.csv", index=False)

    config_summary = {
        "run_name": args.run_name,
        "window_size": int(args.window_size),
        "pca_rank": int(args.pca_rank),
        "ridge_alpha": float(args.ridge_alpha),
        "targets": [int(subject) for subject in target_subjects],
        **summary,
    }
    _save_json(config_summary, root_dir / "summary.json")

    _plot_rho_vs_accuracy(trajectories_by_subject, figures_dir / "rho_vs_window_accuracy.pdf")
    _plot_rho_vs_delta(window_metrics, figures_dir / "rho_vs_delta_scatter.pdf")
    _plot_rho_distribution(window_metrics, figures_dir / "rho_distribution_by_label.pdf")

    logger.info("Saved KCAR diagnostic outputs to %s", root_dir)


if __name__ == "__main__":
    main()
