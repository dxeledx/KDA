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
from src.evaluation.ksda_v3 import load_ksda_v3_folds  # noqa: E402
from src.evaluation.ksda_v31 import (  # noqa: E402
    build_causal_teacher_actions,
    compute_window_oracle_for_actions,
    load_trial_safe_fold_state,
    oracle_usage_stats,
    paired_wins,
    teacher_agreement,
    trialize_window_actions,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d1p5")
TEACHER_WINDOW = 16


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


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d1t-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    d1t_run_dir = _resolve_run_dir(args.d1t_run_dir, "results/ksda/exp_d1t")
    d1t_summary = json.loads((d1t_run_dir / "summary.json").read_text(encoding="utf-8"))
    if (not bool(d1t_summary["trial_safe_action_space_valid"])) and (not args.force):
        raise RuntimeError("D1-T failed; refusing to run D1.5 without --force.")

    rep_cfg = d1t_summary["representation"]
    best_single_action = str(d1t_summary["best_single_action"])

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

    root_dir = ensure_dir(f"results/ksda/exp_d1p5/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)
    trial_label_rows = []
    summary_rows = []
    agreement_rows = []
    histogram_rows = []
    for fold in folds:
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold.cov_source)
        state = load_trial_safe_fold_state(fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)
        action_predictions = {
            name: np.asarray(payload["y_pred"], dtype=np.int64)
            for name, payload in state["action_results"].items()
        }
        action_names = list(action_predictions.keys())
        action_to_idx = {name: idx for idx, name in enumerate(action_names)}

        window_oracle = compute_window_oracle_for_actions(
            np.asarray(fold.y_target_test, dtype=np.int64),
            action_predictions,
            window_size=TEACHER_WINDOW,
        )
        window_oracle_indices = np.asarray(window_oracle["oracle_action_indices"], dtype=np.int64)
        teacher_indices = build_causal_teacher_actions(window_oracle_indices, default_action=action_to_idx["A0_identity"])
        teacher_trial_indices = trialize_window_actions(teacher_indices, len(fold.y_target_test), TEACHER_WINDOW)
        window_oracle_trial_indices = trialize_window_actions(window_oracle_indices, len(fold.y_target_test), TEACHER_WINDOW)

        teacher_pred = np.zeros_like(fold.y_target_test, dtype=np.int64)
        window_oracle_pred = np.zeros_like(fold.y_target_test, dtype=np.int64)
        for action_name, action_idx in action_to_idx.items():
            teacher_mask = teacher_trial_indices == action_idx
            window_mask = window_oracle_trial_indices == action_idx
            teacher_pred[teacher_mask] = action_predictions[action_name][teacher_mask]
            window_oracle_pred[window_mask] = action_predictions[action_name][window_mask]

        best_single_pred = action_predictions[best_single_action]
        teacher_metrics = compute_metrics(fold.y_target_test, teacher_pred)
        best_metrics = compute_metrics(fold.y_target_test, best_single_pred)
        oracle_metrics = compute_metrics(fold.y_target_test, window_oracle_pred)
        summary_rows.append(
            {
                "target_subject": int(fold.target_subject),
                "teacher_accuracy": teacher_metrics["accuracy"],
                "best_single_accuracy": best_metrics["accuracy"],
                "window_oracle_accuracy": oracle_metrics["accuracy"],
            }
        )
        agreement_rows.append(
            {
                "target_subject": int(fold.target_subject),
                "teacher_vs_window_oracle_agreement": teacher_agreement(window_oracle_trial_indices, teacher_trial_indices),
            }
        )
        for action_name, action_idx in action_to_idx.items():
            histogram_rows.append(
                {
                    "target_subject": int(fold.target_subject),
                    "action": action_name,
                    "count": int(np.sum(teacher_trial_indices == action_idx)),
                    "ratio": float(np.mean(teacher_trial_indices == action_idx)),
                }
            )

        for trial_idx in range(len(fold.y_target_test)):
            trial_label_rows.append(
                {
                    "target_subject": int(fold.target_subject),
                    "trial_index": int(trial_idx),
                    "window_id": int(trial_idx // TEACHER_WINDOW),
                    "teacher_action_idx": int(teacher_trial_indices[trial_idx]),
                    "teacher_action_name": action_names[int(teacher_trial_indices[trial_idx])],
                    "window_oracle_action_idx": int(window_oracle_trial_indices[trial_idx]),
                    "window_oracle_action_name": action_names[int(window_oracle_trial_indices[trial_idx])],
                    "y_true": int(fold.y_target_test[trial_idx]),
                    "best_single_pred": int(best_single_pred[trial_idx]),
                    "trialized_oracle_pred": int(teacher_pred[trial_idx]),
                    "window_oracle_pred": int(window_oracle_pred[trial_idx]),
                }
            )

        np.savez(
            details_dir / f"subject_A{int(fold.target_subject):02d}.npz",
            trial_index=np.arange(len(fold.y_target_test), dtype=np.int64),
            window_id=np.arange(len(fold.y_target_test), dtype=np.int64) // TEACHER_WINDOW,
            teacher_action_t=teacher_trial_indices,
            window_oracle_action=window_oracle_trial_indices,
            best_single_action_pred=best_single_pred.astype(np.int64),
            trialized_oracle_pred=teacher_pred.astype(np.int64),
            window_oracle_pred=window_oracle_pred.astype(np.int64),
            y_true=np.asarray(fold.y_target_test, dtype=np.int64),
            action_names=np.asarray(action_names, dtype=object),
        )

    elapsed = float(time.perf_counter() - start)
    trial_labels_df = pd.DataFrame(trial_label_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("target_subject").reset_index(drop=True)
    agreement_df = pd.DataFrame(agreement_rows).sort_values("target_subject").reset_index(drop=True)
    hist_df = pd.DataFrame(histogram_rows)

    teacher_vs_best = paired_wins(summary_df["teacher_accuracy"], summary_df["best_single_accuracy"])
    total_hist = hist_df.groupby("action")["count"].sum().reset_index()
    total_count = max(int(total_hist["count"].sum()), 1)
    total_hist["ratio"] = total_hist["count"] / total_count
    usage = oracle_usage_stats(np.repeat(total_hist["action"].to_numpy(dtype=object), total_hist["count"].to_numpy(dtype=np.int64)))
    positive = np.clip(summary_df["teacher_accuracy"].to_numpy(dtype=np.float64) - summary_df["best_single_accuracy"].to_numpy(dtype=np.float64), 0.0, None)
    max_contrib = float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0
    mean_agreement = float(agreement_df["teacher_vs_window_oracle_agreement"].mean())
    passed = bool(
        (
            teacher_vs_best["mean_delta"] >= 0.01
            or teacher_vs_best["mean_delta"] >= 0.005
        )
        and teacher_vs_best["wins"] >= 6
        and usage["most_common_ratio"] < 0.8
        and max_contrib <= 0.4
        and mean_agreement >= 0.7
    )

    trial_labels_df.to_csv(root_dir / "causal_trial_labels.csv", index=False)
    agreement_df.to_csv(root_dir / "teacher_vs_window_oracle_agreement.csv", index=False)
    total_hist.to_csv(root_dir / "teacher_action_histogram.csv", index=False)
    summary_table = pd.DataFrame(
        [
            {
                "teacher_window_size": TEACHER_WINDOW,
                "teacher_accuracy_mean": float(summary_df["teacher_accuracy"].mean()),
                "best_single_action": best_single_action,
                "best_single_accuracy_mean": float(summary_df["best_single_accuracy"].mean()),
                "window_oracle_accuracy_mean": float(summary_df["window_oracle_accuracy"].mean()),
                "mean_delta_vs_best_single": float(teacher_vs_best["mean_delta"]),
                "wins_vs_best_single": int(teacher_vs_best["wins"]),
                "losses_vs_best_single": int(teacher_vs_best["losses"]),
                "draws_vs_best_single": int(teacher_vs_best["draws"]),
                "most_common_ratio": float(usage["most_common_ratio"]),
                "max_single_subject_contribution": max_contrib,
                "teacher_vs_window_oracle_agreement": mean_agreement,
                "passed": passed,
            }
        ]
    )
    summary_table.to_csv(root_dir / "trialized_oracle_summary.csv", index=False)
    summary = {
        "d1t_run_dir": str(d1t_run_dir),
        "representation": rep_cfg,
        "teacher_window_size": TEACHER_WINDOW,
        "best_single_action": best_single_action,
        "causal_trialization_valid": passed,
        "trialized_oracle_summary": summary_table.iloc[0].to_dict(),
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plot_df = pd.DataFrame(
        {
            "method": ["best_single_action", "causal_trialized_oracle", "window_oracle"],
            "accuracy": [
                float(summary_df["best_single_accuracy"].mean()),
                float(summary_df["teacher_accuracy"].mean()),
                float(summary_df["window_oracle_accuracy"].mean()),
            ],
        }
    )
    plt.bar(plot_df["method"], plot_df["accuracy"])
    plt.ylabel("Accuracy")
    plt.title("D1.5 causal trialization")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "trialized_oracle_comparison.pdf", dpi=300)
    plt.close()

    logger.info("Saved D1.5 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
