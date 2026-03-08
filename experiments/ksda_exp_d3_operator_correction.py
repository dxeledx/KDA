#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ksda_exp_d2_linear_performance_proxy import (  # noqa: E402
    _apply_window_selector,
    _build_fold_state,
    _feature_matrix_for_fold,
    _load_custom_fold,
    _resolve_targets,
    _window_oracle_labels,
)
from src.evaluation.kcar_analysis import KoopmanOperator, compute_transition_residuals  # noqa: E402
from src.evaluation.ksda_v3 import (  # noqa: E402
    load_ksda_v3_folds,
    paired_wins,
    predict_linear_multiclass_selector,
    predict_linear_scalar_proxy,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d3")
GAMMA_CANDIDATES = [0.05, 0.1]


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _resolve_run_dir(run_dir: str | None, root: str) -> Path:
    base = Path(root)
    if run_dir:
        return Path(run_dir)
    candidates = sorted([path for path in base.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directory found under {base}")
    return candidates[-1]


def _load_target_models(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return {key: obj[key] for key in obj.files}


def _scalar_model_from_npz(model_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "mean": np.asarray(model_npz["gain_mean"], dtype=np.float64),
        "std": np.asarray(model_npz["gain_std"], dtype=np.float64),
        "beta": np.asarray(model_npz["gain_beta"], dtype=np.float64),
    }


def _selector_model_from_npz(model_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "mean": np.asarray(model_npz["selector_mean"], dtype=np.float64),
        "std": np.asarray(model_npz["selector_std"], dtype=np.float64),
        "weights": np.asarray(model_npz["selector_weights"], dtype=np.float64),
    }


def _correct_feature_windows(
    features: np.ndarray,
    z_states: np.ndarray,
    source_operator,
    target_operator,
    gain_model: Dict[str, np.ndarray],
    gamma: float,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta_k = np.asarray(target_operator.matrix - source_operator.matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(delta_k, full_matrices=False)
    U = u[:, [0]]
    V = vh[:1, :]

    corrected = np.asarray(features, dtype=np.float64).copy()
    corrected_r_tgt = []
    for window_idx, (start, end) in enumerate(_window_slices(len(z_states), window_size)):
        score = float(predict_linear_scalar_proxy(gain_model, features[window_idx : window_idx + 1])[0])
        alpha = float(np.clip(float(gamma) * score, -0.1, 0.1))
        k_tilde = KoopmanOperator(matrix=np.asarray(target_operator.matrix + alpha * (U @ V), dtype=np.float64))
        states = z_states[start:end]
        if len(states) >= 2:
            r_src = float(np.mean(compute_transition_residuals(states, source_operator)))
            r_tgt = float(np.mean(compute_transition_residuals(states, k_tilde)))
        else:
            r_src = 0.0
            r_tgt = 0.0
        corrected[window_idx, 0] = r_src
        corrected[window_idx, 1] = r_tgt
        corrected[window_idx, 2] = r_tgt - r_src
        corrected_r_tgt.append(r_tgt)
    return corrected, np.asarray(corrected_r_tgt, dtype=np.float64)


def _window_slices(total_trials: int, window_size: int) -> List[tuple[int, int]]:
    return [(start, min(start + window_size, total_trials)) for start in range(0, total_trials, window_size)]


def _choose_gamma(
    loader,
    all_subjects: List[int],
    actual_target: int,
    rep_cfg: Dict[str, object],
    window_size: int,
    lda_kwargs: Dict[str, object],
    cov_eps: float,
    pre,
    selector_model: Dict[str, np.ndarray],
    gain_model: Dict[str, np.ndarray],
) -> float:
    source_subjects = [subject for subject in all_subjects if subject != actual_target]
    gamma_scores = {gamma: [] for gamma in GAMMA_CANDIDATES}
    for pseudo_target in source_subjects:
        pseudo_sources = [subject for subject in source_subjects if subject != pseudo_target]
        pseudo_fold = _load_custom_fold(loader, pseudo_sources, pseudo_target, pre, cov_eps)
        pseudo_state = _build_fold_state(pseudo_fold, rep_cfg, lda_kwargs, cov_eps)
        feature_bundle = _feature_matrix_for_fold(pseudo_fold, pseudo_state, window_size)
        oracle = _window_oracle_labels(pseudo_fold, pseudo_state, window_size)
        oracle_labels = np.asarray([row["label_idx"] for row in oracle["rows"]], dtype=np.int64)

        for gamma in GAMMA_CANDIDATES:
            corrected, _ = _correct_feature_windows(
                feature_bundle["features"],
                pseudo_state["z_target_test"],
                pseudo_state["source_operator"],
                pseudo_state["target_operator"],
                gain_model,
                gamma,
                window_size,
            )
            chosen, _ = predict_linear_multiclass_selector(selector_model, corrected)
            gamma_scores[gamma].append(float(np.mean(chosen == oracle_labels)))

    ranked = sorted(gamma_scores.items(), key=lambda item: (np.mean(item[1]), -item[0]), reverse=True)
    return float(ranked[0][0])


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d2-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    d2_run_dir = _resolve_run_dir(args.d2_run_dir, "results/ksda/exp_d2")
    d2_summary = json.loads((d2_run_dir / "summary.json").read_text(encoding="utf-8"))
    if (not bool(d2_summary["passed"])) and (not args.force):
        raise RuntimeError("D2 did not pass; refusing to run D3 without --force.")

    rep_cfg = d2_summary["representation"]
    window_size = int(d2_summary["window_size"])
    best_single_expert = str(d2_summary["best_single_expert"])

    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))
    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    all_subjects = list(map(int, data_cfg["dataset"]["subjects"]))
    target_subjects = _resolve_targets(all_subjects, args.targets)
    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/ksda/exp_d3/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)
    summary_rows = []
    gamma_rows = []
    residual_rows = []

    for fold in folds:
        target = int(fold.target_subject)
        model_npz = _load_target_models(d2_run_dir / "models" / f"target_A{target:02d}.npz")
        selector_model = _selector_model_from_npz(model_npz)
        gain_model = _scalar_model_from_npz(model_npz)
        gamma = _choose_gamma(loader, all_subjects, target, rep_cfg, window_size, lda_kwargs, cov_eps, pre, selector_model, gain_model)
        gamma_rows.append({"target_subject": target, "gamma": gamma})

        state = _build_fold_state(fold, rep_cfg, lda_kwargs, cov_eps)
        feature_bundle = _feature_matrix_for_fold(fold, state, window_size)
        base_chosen, _ = predict_linear_multiclass_selector(selector_model, feature_bundle["features"])
        y_pred_base = _apply_window_selector(state["expert_predictions"], state["expert_names"], base_chosen, len(fold.y_target_test), window_size)

        corrected_features, corrected_r_tgt = _correct_feature_windows(
            feature_bundle["features"],
            state["z_target_test"],
            state["source_operator"],
            state["target_operator"],
            gain_model,
            gamma,
            window_size,
        )
        corrected_chosen, _ = predict_linear_multiclass_selector(selector_model, corrected_features)
        y_pred_corrected = _apply_window_selector(state["expert_predictions"], state["expert_names"], corrected_chosen, len(fold.y_target_test), window_size)
        y_pred_best = state["expert_predictions"][best_single_expert]

        base_metrics = compute_metrics(fold.y_target_test, y_pred_base)
        corrected_metrics = compute_metrics(fold.y_target_test, y_pred_corrected)
        best_metrics = compute_metrics(fold.y_target_test, y_pred_best)
        summary_rows.extend(
            [
                {"method": "d2_selector", "target_subject": target, **base_metrics},
                {"method": "d3_corrected_selector", "target_subject": target, **corrected_metrics},
                {"method": "best_single_expert", "target_subject": target, **best_metrics},
            ]
        )

        residual_rows.append(
            {
                "target_subject": target,
                "base_r_tgt_mean": float(np.mean(feature_bundle["features"][:, 1])),
                "corrected_r_tgt_mean": float(np.mean(corrected_r_tgt)),
                "residual_delta": float(np.mean(corrected_r_tgt) - np.mean(feature_bundle["features"][:, 1])),
            }
        )
        np.savez(
            details_dir / f"subject_A{target:02d}.npz",
            y_true=np.asarray(fold.y_target_test, dtype=np.int64),
            chosen_base=base_chosen,
            chosen_corrected=corrected_chosen,
            features_base=feature_bundle["features"],
            features_corrected=corrected_features,
            trial_index=np.arange(len(fold.y_target_test), dtype=np.int64),
            window_id=feature_bundle["window_id"],
        )

    elapsed = float(time.perf_counter() - start)
    summary_df = pd.DataFrame(summary_rows)
    residual_df = pd.DataFrame(residual_rows).sort_values("target_subject").reset_index(drop=True)
    gamma_df = pd.DataFrame(gamma_rows).sort_values("target_subject").reset_index(drop=True)

    base_df = summary_df.loc[summary_df["method"] == "d2_selector"].sort_values("target_subject")
    corrected_df = summary_df.loc[summary_df["method"] == "d3_corrected_selector"].sort_values("target_subject")
    best_df = summary_df.loc[summary_df["method"] == "best_single_expert"].sort_values("target_subject")

    corrected_vs_base = paired_wins(corrected_df["accuracy"], base_df["accuracy"])
    corrected_vs_best = paired_wins(corrected_df["accuracy"], best_df["accuracy"])
    worst_delta = float(np.min(corrected_df["accuracy"].to_numpy(dtype=np.float64) - base_df["accuracy"].to_numpy(dtype=np.float64)))
    residual_improved = bool(float(residual_df["residual_delta"].mean()) < 0.0)
    passed = bool(
        residual_improved
        and corrected_vs_base["mean_delta"] >= 0.005
        and worst_delta >= -0.02
    )

    summary_table = (
        summary_df.groupby("method")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    )
    summary_table.to_csv(root_dir / "correction_summary.csv", index=False)
    residual_df.to_csv(root_dir / "residual_summary.csv", index=False)
    gamma_df.to_csv(root_dir / "gamma_summary.csv", index=False)
    summary_df.to_csv(root_dir / "correction_loso.csv", index=False)

    summary = {
        "d2_run_dir": str(d2_run_dir),
        "representation": rep_cfg,
        "window_size": window_size,
        "best_single_expert": best_single_expert,
        "pairwise": {
            "corrected_vs_d2_selector": corrected_vs_base,
            "corrected_vs_best_single": corrected_vs_best,
        },
        "mean_residual_delta": float(residual_df["residual_delta"].mean()),
        "worst_subject_delta_vs_d2": worst_delta,
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plot_df = summary_table.set_index("method").loc[
        ["best_single_expert", "d2_selector", "d3_corrected_selector"]
    ].reset_index()
    plt.bar(plot_df["method"], plot_df["accuracy_mean"])
    plt.ylabel("Accuracy")
    plt.title("D3 operator correction comparison")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "operator_correction_comparison.pdf", dpi=300)
    plt.close()

    logger.info("Saved D3 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
