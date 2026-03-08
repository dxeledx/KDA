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
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.ksda_v3 import (  # noqa: E402
    build_temporary_endpoint_aligner,
    build_window_slices,
    compute_window_oracle_actions,
    load_ksda_v3_folds,
    oracle_usage_stats,
    paired_wins,
)
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d0")

DEFAULT_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_WINDOW_SIZES = [16, 32, 48]


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


def _resolve_run_dir(run_dir: str | None) -> Path:
    root = Path("results/ksda/exp_d1r")
    if run_dir:
        return Path(run_dir)
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError("No D1-R run directory found under results/ksda/exp_d1r")
    return candidates[-1]


def _parse_float_list(raw: str, cast=float) -> List[float]:
    return [cast(token.strip()) for token in raw.split(",") if token.strip()]


def _resolve_targets(all_subjects: List[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def _evaluate_fold(
    fold,
    rep_cfg: Dict[str, object],
    best_static_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
    cov_eps: float,
    weights: List[float],
    window_sizes: List[int],
) -> Dict[str, object]:
    projector = KoopmanFeatureProjector(
        pca_rank=int(rep_cfg["pca_rank"]),
        lifting=str(rep_cfg["lifting"]),
    ).fit(fold.cov_source)

    psi_source = projector.transform(fold.cov_source)
    psi_target_train = projector.transform(fold.cov_target_train)
    psi_target_test = projector.transform(fold.cov_target_test)
    endpoint = build_temporary_endpoint_aligner(
        str(best_static_cfg["method"]),
        psi_source,
        psi_target_train,
        fold.y_source,
        k=best_static_cfg.get("k"),
        reg_lambda=best_static_cfg.get("reg_lambda"),
        normalize_output=bool(best_static_cfg.get("normalize_output", False)),
        cov_eps=cov_eps,
    )

    psi_source_endpoint = endpoint.transform(psi_source)
    psi_target_endpoint = endpoint.transform(psi_target_test)
    lda = LDA(**lda_kwargs).fit(psi_source_endpoint, fold.y_source)

    trial_preds = {}
    fixed_subject_rows = []
    detail_rows = []
    oracle_subject_rows = []
    histogram_rows = []
    for w in weights:
        psi_test_w = (1.0 - w) * psi_target_test + float(w) * psi_target_endpoint
        y_pred = lda.predict(psi_test_w).astype(np.int64)
        trial_preds[float(w)] = y_pred

    for window_size in window_sizes:
        window_scores = []
        window_actions = []
        oracle_predictions = np.zeros_like(fold.y_target_test, dtype=np.int64)
        for window_idx, (start, end) in enumerate(build_window_slices(len(fold.y_target_test), int(window_size))):
            y_window = fold.y_target_test[start:end]
            window_accs = []
            for w in weights:
                y_pred_window = trial_preds[float(w)][start:end]
                window_accs.append(float(np.mean(y_pred_window == y_window)))
            window_accs_array = np.asarray(window_accs, dtype=np.float64)
            action_idx, _ = compute_window_oracle_actions(window_accs_array[None, :])
            best_idx = int(action_idx[0])
            best_w = float(weights[best_idx])
            window_scores.append(window_accs_array)
            window_actions.append(best_w)
            oracle_predictions[start:end] = trial_preds[best_w][start:end]
            detail_rows.append(
                {
                    "window_size": int(window_size),
                    "window_id": int(window_idx),
                    "oracle_w": best_w,
                    **{f"acc_w_{str(w).replace('.', '_')}": float(acc) for w, acc in zip(weights, window_accs_array)},
                }
            )

        oracle_metrics = compute_metrics(fold.y_target_test, oracle_predictions)
        oracle_subject_rows.append(
            {
                "target_subject": int(fold.target_subject),
                "window_size": int(window_size),
                **oracle_metrics,
            }
        )
        usage = oracle_usage_stats(np.asarray(window_actions, dtype=np.float64))
        for action_value in weights:
            histogram_rows.append(
                {
                    "target_subject": int(fold.target_subject),
                    "window_size": int(window_size),
                    "w": float(action_value),
                    "count": int(np.sum(np.isclose(window_actions, action_value))),
                    "ratio": float(np.mean(np.isclose(window_actions, action_value))),
                }
            )
        for w in weights:
            fixed_subject_rows.append(
                {
                    "target_subject": int(fold.target_subject),
                    "window_size": int(window_size),
                    "w": float(w),
                    **compute_metrics(fold.y_target_test, trial_preds[float(w)]),
                }
            )

    return {
        "target_subject": int(fold.target_subject),
        "y_true": np.asarray(fold.y_target_test, dtype=np.int64),
        "trial_index": np.arange(len(fold.y_target_test), dtype=np.int64),
        "fixed_subject_rows": fixed_subject_rows,
        "oracle_subject_rows": oracle_subject_rows,
        "histogram_rows": histogram_rows,
        "detail_rows": detail_rows,
        "trial_preds": trial_preds,
    }


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d1r-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--window-sizes", default="16,32,48")
    args = parser.parse_args()

    d1r_run_dir = _resolve_run_dir(args.d1r_run_dir)
    best_static_info = json.loads((d1r_run_dir / "best_static.json").read_text(encoding="utf-8"))
    rep_cfg = best_static_info["representation"]
    best_static_cfg = best_static_info["best_static"]

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
    weights = _parse_float_list(args.weights, cast=float)
    window_sizes = [int(v) for v in _parse_float_list(args.window_sizes, cast=int)]

    root_dir = ensure_dir(f"results/ksda/exp_d0/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)
    fixed_rows, oracle_rows, histogram_rows = [], [], []
    for fold in folds:
        fold_result = _evaluate_fold(
            fold,
            rep_cfg,
            best_static_cfg,
            lda_kwargs,
            cov_eps,
            weights,
            window_sizes,
        )
        fixed_rows.extend(fold_result["fixed_subject_rows"])
        oracle_rows.extend(fold_result["oracle_subject_rows"])
        histogram_rows.extend(fold_result["histogram_rows"])

        detail_df = pd.DataFrame(fold_result["detail_rows"])
        save_dict = {
            "y_true": fold_result["y_true"],
            "trial_index": fold_result["trial_index"],
            "window_size": detail_df["window_size"].to_numpy(dtype=np.int64),
            "window_id": detail_df["window_id"].to_numpy(dtype=np.int64),
            "oracle_w": detail_df["oracle_w"].to_numpy(dtype=np.float64),
        }
        for w in weights:
            key = f"acc_w_{str(w).replace('.', '_')}"
            save_dict[key] = detail_df[key].to_numpy(dtype=np.float64)
        np.savez(details_dir / f"subject_A{int(fold.target_subject):02d}.npz", **save_dict)

    elapsed = float(time.perf_counter() - start)
    fixed_df = pd.DataFrame(fixed_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    histogram_df = pd.DataFrame(histogram_rows)

    fixed_summary_rows = []
    oracle_summary_rows = []
    for window_size in window_sizes:
        fixed_ws = fixed_df.loc[fixed_df["window_size"] == window_size].copy()
        oracle_ws = oracle_df.loc[oracle_df["window_size"] == window_size].copy()
        fixed_agg = (
            fixed_ws.groupby("w")["accuracy"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
        )
        for _, row in fixed_agg.iterrows():
            fixed_summary_rows.append(
                {
                    "window_size": int(window_size),
                    "w": float(row["w"]),
                    "accuracy_mean": float(row["accuracy_mean"]),
                    "accuracy_std": float(row["accuracy_std"]) if pd.notna(row["accuracy_std"]) else 0.0,
                }
            )

        best_fixed_row = fixed_agg.sort_values(["accuracy_mean", "w"], ascending=[False, True]).iloc[0]
        best_fixed_w = float(best_fixed_row["w"])
        best_fixed_subject = fixed_ws.loc[fixed_ws["w"] == best_fixed_w].sort_values("target_subject")
        oracle_subject = oracle_ws.sort_values("target_subject")
        pair = paired_wins(oracle_subject["accuracy"], best_fixed_subject["accuracy"])

        hist_ws = histogram_df.loc[histogram_df["window_size"] == window_size]
        total_count_per_w = hist_ws.groupby("w")["count"].sum().reset_index()
        total_actions = max(int(total_count_per_w["count"].sum()), 1)
        hist_stats = oracle_usage_stats(np.repeat(total_count_per_w["w"].to_numpy(dtype=np.float64), total_count_per_w["count"].to_numpy(dtype=np.int64)))
        distinct_subjects = (
            hist_ws.groupby("target_subject")["w"].nunique().reset_index(name="unique_w")
        )
        subjects_with_multiple = int(np.sum(distinct_subjects["unique_w"].to_numpy(dtype=np.int64) >= 2))
        passed = bool(
            (
                pair["mean_delta"] >= 0.01
                or (pair["mean_delta"] >= 0.005 and pair["wins"] >= 6)
            )
            and hist_stats["most_common_ratio"] < 0.8
            and subjects_with_multiple >= 6
        )
        oracle_summary_rows.append(
            {
                "window_size": int(window_size),
                "oracle_accuracy_mean": float(oracle_subject["accuracy"].mean()),
                "best_fixed_w": best_fixed_w,
                "best_fixed_accuracy_mean": float(best_fixed_row["accuracy_mean"]),
                "mean_delta_vs_best_fixed": float(pair["mean_delta"]),
                "wins_vs_best_fixed": int(pair["wins"]),
                "losses_vs_best_fixed": int(pair["losses"]),
                "draws_vs_best_fixed": int(pair["draws"]),
                "most_common_ratio": float(hist_stats["most_common_ratio"]),
                "subjects_with_multiple_actions": subjects_with_multiple,
                "passed": passed,
            }
        )

    fixed_summary_df = pd.DataFrame(fixed_summary_rows).sort_values(["window_size", "accuracy_mean"], ascending=[True, False])
    oracle_summary_df = pd.DataFrame(oracle_summary_rows).sort_values(["passed", "oracle_accuracy_mean"], ascending=[False, False])
    histogram_summary = (
        histogram_df.groupby(["window_size", "w"])["count"]
        .sum()
        .reset_index()
    )
    histogram_summary["ratio"] = histogram_summary.groupby("window_size")["count"].transform(lambda col: col / max(int(col.sum()), 1))

    fixed_summary_df.to_csv(root_dir / "fixed_w_summary.csv", index=False)
    oracle_summary_df.to_csv(root_dir / "window_oracle_summary.csv", index=False)
    histogram_summary.to_csv(root_dir / "oracle_action_histogram.csv", index=False)
    fixed_df.to_csv(root_dir / "fixed_w_loso.csv", index=False)
    oracle_df.to_csv(root_dir / "oracle_loso.csv", index=False)

    best_row = oracle_summary_df.iloc[0]
    summary = {
        "d1r_run_dir": str(d1r_run_dir),
        "weights": weights,
        "window_sizes": window_sizes,
        "best_window_size": int(best_row["window_size"]),
        "dynamic_need_exists": bool(np.any(oracle_summary_df["passed"].to_numpy(dtype=bool))),
        "best_window_summary": best_row.to_dict(),
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.5, 4.5))
    for window_size in window_sizes:
        subset = fixed_summary_df.loc[fixed_summary_df["window_size"] == window_size].sort_values("w")
        plt.plot(subset["w"], subset["accuracy_mean"], marker="o", label=f"fixed, win={window_size}")
        oracle_acc = float(oracle_summary_df.loc[oracle_summary_df["window_size"] == window_size, "oracle_accuracy_mean"].iloc[0])
        plt.axhline(oracle_acc, linestyle="--", alpha=0.4)
    plt.xlabel("w")
    plt.ylabel("Accuracy")
    plt.title("D0 fixed-w vs oracle-window")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fixed_vs_oracle.pdf", dpi=300)
    plt.close()

    logger.info("Saved D0 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
