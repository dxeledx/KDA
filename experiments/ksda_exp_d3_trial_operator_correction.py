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

from src.alignment.koopman_alignment import KoopmanFeatureProjector  # noqa: E402
from src.evaluation.kcar_analysis import KoopmanOperator, compute_transition_residuals  # noqa: E402
from src.evaluation.ksda_v31 import (  # noqa: E402
    build_causal_trialized_teacher_from_state,
    compute_trial_features,
    fit_linear_scalar_proxy,
    load_custom_ksda_fold,
    load_trial_safe_fold_state,
    paired_wins,
    predict_linear_multiclass_selector,
    predict_linear_scalar_proxy,
)
from src.evaluation.ksda_v3 import load_ksda_v3_folds  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d3_trial")
TRAILING_LEN = 16
TEACHER_WINDOW = 16
UPDATE_EVERY = 8
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


def _resolve_targets(all_subjects: List[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def _load_selector_model(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return {key: obj[key] for key in obj.files}


def _apply_trial_action_selector(
    action_predictions: Dict[str, np.ndarray],
    action_names: List[str],
    chosen_labels: np.ndarray,
) -> np.ndarray:
    chosen_labels = np.asarray(chosen_labels, dtype=np.int64)
    total_trials = len(next(iter(action_predictions.values())))
    y_pred = np.zeros(total_trials, dtype=np.int64)
    for idx, action_name in enumerate(action_names):
        mask = chosen_labels == idx
        y_pred[mask] = action_predictions[action_name][mask]
    return y_pred


def _fit_scalar_proxy_for_target(
    loader,
    all_subjects: List[int],
    actual_target: int,
    rep_cfg: Dict[str, object],
    best_single_action: str,
    lda_kwargs: Dict[str, object],
    cov_eps: float,
    pre,
) -> Dict[str, np.ndarray]:
    X_blocks = []
    y_blocks = []
    source_subjects = [subject for subject in all_subjects if subject != actual_target]
    for pseudo_target in source_subjects:
        pseudo_sources = [subject for subject in source_subjects if subject != pseudo_target]
        pseudo_fold = load_custom_ksda_fold(loader, pseudo_sources, pseudo_target, pre, cov_eps)
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(pseudo_fold.cov_source)
        state = load_trial_safe_fold_state(pseudo_fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)
        teacher = build_causal_trialized_teacher_from_state(pseudo_fold, state, window_size=TEACHER_WINDOW)
        features = compute_trial_features(
            state["z_target_test"],
            state["psi_target_test"],
            state["source_operator"],
            state["target_operator"],
            state["raw_lda"],
            trailing_len=TRAILING_LEN,
            source_mean_z=state["source_mean_z"],
            source_diag_var_z=state["source_diag_var_z"],
        )

        teacher_action_names = np.asarray(teacher["action_names"], dtype=object)
        teacher_pred = np.asarray(teacher["teacher_pred"], dtype=np.int64)
        best_pred = np.asarray(state["action_results"][best_single_action]["y_pred"], dtype=np.int64)
        gain = (teacher_pred == pseudo_fold.y_target_test).astype(np.float64) - (best_pred == pseudo_fold.y_target_test).astype(np.float64)
        smooth_gain = np.zeros_like(gain, dtype=np.float64)
        for t in range(len(gain)):
            start = max(0, t - TRAILING_LEN + 1)
            smooth_gain[t] = float(np.mean(gain[start : t + 1]))

        X_blocks.append(features)
        y_blocks.append(smooth_gain)

    X = np.concatenate(X_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)
    return fit_linear_scalar_proxy(X, y, reg_lambda=1.0e-3)


def _choose_gamma(
    state,
    selector_model: Dict[str, np.ndarray],
    scalar_model: Dict[str, np.ndarray],
    best_single_action: str,
) -> float:
    delta_k = np.asarray(state["target_operator"].matrix - state["source_operator"].matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(delta_k, full_matrices=False)
    U = u[:, [0]]
    V = vh[:1, :]
    base_features = compute_trial_features(
        state["z_target_test"],
        state["psi_target_test"],
        state["source_operator"],
        state["target_operator"],
        state["raw_lda"],
        trailing_len=TRAILING_LEN,
        source_mean_z=state["source_mean_z"],
        source_diag_var_z=state["source_diag_var_z"],
    )
    best_gamma = GAMMA_CANDIDATES[0]
    best_score = -1e9
    for gamma in GAMMA_CANDIDATES:
        features = base_features.copy()
        r_tgt_corrected = []
        current_operator = state["target_operator"]
        for t in range(len(state["z_target_test"])):
            if t > 0 and t % UPDATE_EVERY == 0:
                score = float(predict_linear_scalar_proxy(scalar_model, features[t - 1 : t])[0])
                alpha = float(np.clip(float(gamma) * score, -0.1, 0.1))
                current_operator = KoopmanOperator(
                    matrix=np.asarray(state["target_operator"].matrix + alpha * (U @ V), dtype=np.float64)
                )
            start = max(0, t - TRAILING_LEN + 1)
            z_hist = state["z_target_test"][start : t + 1]
            if len(z_hist) >= 2:
                features[t, 1] = float(np.mean(compute_transition_residuals(z_hist, current_operator)))
                features[t, 2] = float(features[t, 1] - features[t, 0])
            r_tgt_corrected.append(float(features[t, 1]))
        chosen, _ = predict_linear_multiclass_selector(selector_model, features)
        y_pred = _apply_trial_action_selector(
            {name: np.asarray(payload["y_pred"], dtype=np.int64) for name, payload in state["action_results"].items()},
            state["action_names"],
            chosen,
        )
        score = float(np.mean(y_pred == state["y_true"])) - float(np.mean(r_tgt_corrected))
        if score > best_score:
            best_score = score
            best_gamma = gamma
    return float(best_gamma)


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

    d2_run_dir = _resolve_run_dir(args.d2_run_dir, "results/ksda/exp_d2_trial")
    d2_summary = json.loads((d2_run_dir / "summary.json").read_text(encoding="utf-8"))
    if (not bool(d2_summary["passed"])) and (not args.force):
        raise RuntimeError("D2 did not pass; refusing to run D3 without --force.")

    d1p5_run_dir = Path(d2_summary["d1p5_run_dir"])
    rep_cfg = d2_summary["representation"]
    best_single_action = str(d2_summary["best_single_action"])

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

    root_dir = ensure_dir(f"results/ksda/exp_d3_trial/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)

    rows = []
    residual_rows = []
    gamma_rows = []
    for fold in folds:
        target = int(fold.target_subject)
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold.cov_source)
        state = load_trial_safe_fold_state(fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)
        state["y_true"] = np.asarray(fold.y_target_test, dtype=np.int64)
        teacher = build_causal_trialized_teacher_from_state(fold, state, window_size=TEACHER_WINDOW)
        selector_npz = _load_selector_model(d2_run_dir / "models" / f"target_A{target:02d}.npz")
        selector_model = {
            "mean": selector_npz["mean"],
            "std": selector_npz["std"],
            "weights": selector_npz["weights"],
        }
        scalar_model = _fit_scalar_proxy_for_target(
            loader,
            all_subjects,
            target,
            rep_cfg,
            best_single_action,
            lda_kwargs,
            cov_eps,
            pre,
        )
        gamma = _choose_gamma(state, selector_model, scalar_model, best_single_action)
        gamma_rows.append({"target_subject": target, "gamma": gamma})

        delta_k = np.asarray(state["target_operator"].matrix - state["source_operator"].matrix, dtype=np.float64)
        u, _, vh = np.linalg.svd(delta_k, full_matrices=False)
        U = u[:, [0]]
        V = vh[:1, :]

        base_features = compute_trial_features(
            state["z_target_test"],
            state["psi_target_test"],
            state["source_operator"],
            state["target_operator"],
            state["raw_lda"],
            trailing_len=TRAILING_LEN,
            source_mean_z=state["source_mean_z"],
            source_diag_var_z=state["source_diag_var_z"],
        )
        chosen_base, _ = predict_linear_multiclass_selector(selector_model, base_features)
        y_pred_base = _apply_trial_action_selector(
            {name: np.asarray(payload["y_pred"], dtype=np.int64) for name, payload in state["action_results"].items()},
            state["action_names"],
            chosen_base,
        )

        corrected_features = base_features.copy()
        current_operator = state["target_operator"]
        alpha_series = np.zeros(len(state["y_true"]), dtype=np.float64)
        corrected_r_tgt = []
        for t in range(len(state["y_true"])):
            if t > 0 and t % UPDATE_EVERY == 0:
                score = float(predict_linear_scalar_proxy(scalar_model, corrected_features[t - 1 : t])[0])
                alpha = float(np.clip(gamma * score, -0.1, 0.1))
                alpha_series[t:] = alpha
                current_operator = KoopmanOperator(
                    matrix=np.asarray(state["target_operator"].matrix + alpha * (U @ V), dtype=np.float64)
                )
            start_idx = max(0, t - TRAILING_LEN + 1)
            z_hist = state["z_target_test"][start_idx : t + 1]
            if len(z_hist) >= 2:
                corrected_features[t, 1] = float(np.mean(compute_transition_residuals(z_hist, current_operator)))
                corrected_features[t, 2] = float(corrected_features[t, 1] - corrected_features[t, 0])
            corrected_r_tgt.append(float(corrected_features[t, 1]))

        chosen_corrected, _ = predict_linear_multiclass_selector(selector_model, corrected_features)
        y_pred_corrected = _apply_trial_action_selector(
            {name: np.asarray(payload["y_pred"], dtype=np.int64) for name, payload in state["action_results"].items()},
            state["action_names"],
            chosen_corrected,
        )
        y_pred_teacher = np.asarray(teacher["teacher_pred"], dtype=np.int64)
        y_pred_best = np.asarray(state["action_results"][best_single_action]["y_pred"], dtype=np.int64)

        rows.extend(
            [
                {"method": "d2_selector", "target_subject": target, **compute_metrics(state["y_true"], y_pred_base)},
                {"method": "d3_corrected_selector", "target_subject": target, **compute_metrics(state["y_true"], y_pred_corrected)},
                {"method": "best_single_action", "target_subject": target, **compute_metrics(state["y_true"], y_pred_best)},
                {"method": "causal_trialized_oracle", "target_subject": target, **compute_metrics(state["y_true"], y_pred_teacher)},
            ]
        )
        residual_rows.append(
            {
                "target_subject": target,
                "base_r_tgt_mean": float(np.mean(base_features[:, 1])),
                "corrected_r_tgt_mean": float(np.mean(corrected_r_tgt)),
                "residual_delta": float(np.mean(corrected_r_tgt) - np.mean(base_features[:, 1])),
                "alpha_mean": float(np.mean(alpha_series)),
                "alpha_max_abs": float(np.max(np.abs(alpha_series))),
            }
        )
        np.savez(
            details_dir / f"subject_A{target:02d}.npz",
            trial_index=np.arange(len(state["y_true"]), dtype=np.int64),
            y_true=np.asarray(state["y_true"], dtype=np.int64),
            chosen_base=chosen_base,
            chosen_corrected=chosen_corrected,
            y_pred_base=y_pred_base,
            y_pred_corrected=y_pred_corrected,
            y_pred_best=y_pred_best,
            y_pred_teacher=y_pred_teacher,
            base_features=base_features,
            corrected_features=corrected_features,
            alpha_t=alpha_series,
            action_names=np.asarray(state["action_names"], dtype=object),
        )

    elapsed = float(time.perf_counter() - start)
    rows_df = pd.DataFrame(rows)
    residual_df = pd.DataFrame(residual_rows).sort_values("target_subject").reset_index(drop=True)
    gamma_df = pd.DataFrame(gamma_rows).sort_values("target_subject").reset_index(drop=True)

    base_df = rows_df.loc[rows_df["method"] == "d2_selector"].sort_values("target_subject")
    corrected_df = rows_df.loc[rows_df["method"] == "d3_corrected_selector"].sort_values("target_subject")
    best_df = rows_df.loc[rows_df["method"] == "best_single_action"].sort_values("target_subject")

    corrected_vs_base = paired_wins(corrected_df["accuracy"], base_df["accuracy"])
    corrected_vs_best = paired_wins(corrected_df["accuracy"], best_df["accuracy"])
    worst_delta = float(np.min(corrected_df["accuracy"].to_numpy(dtype=np.float64) - base_df["accuracy"].to_numpy(dtype=np.float64)))
    passed = bool(
        float(residual_df["residual_delta"].mean()) < 0.0
        and corrected_vs_base["mean_delta"] >= 0.005
        and worst_delta >= -0.02
    )

    rows_df.to_csv(root_dir / "correction_loso.csv", index=False)
    residual_df.to_csv(root_dir / "residual_summary.csv", index=False)
    gamma_df.to_csv(root_dir / "gamma_summary.csv", index=False)
    summary_df = (
        rows_df.groupby("method")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    )
    summary_df.to_csv(root_dir / "correction_summary.csv", index=False)

    summary = {
        "d2_run_dir": str(d2_run_dir),
        "representation": rep_cfg,
        "best_single_action": best_single_action,
        "pairwise": {
            "corrected_vs_d2": corrected_vs_base,
            "corrected_vs_best_single": corrected_vs_best,
        },
        "mean_residual_delta": float(residual_df["residual_delta"].mean()),
        "worst_subject_delta_vs_d2": worst_delta,
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plot_df = summary_df.set_index("method").loc[
        ["best_single_action", "d2_selector", "d3_corrected_selector", "causal_trialized_oracle"]
    ].reset_index()
    plt.bar(plot_df["method"], plot_df["accuracy_mean"])
    plt.ylabel("Accuracy")
    plt.title("D3 trial-level operator correction")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(root_dir / "figures" / "operator_correction_comparison.pdf", dpi=300)
    plt.close()

    logger.info("Saved D3 trial outputs to %s", root_dir)


if __name__ == "__main__":
    main()
