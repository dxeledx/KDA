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

from src.evaluation.ksda_v3 import (  # noqa: E402
    apply_expert_aligner,
    build_local_expert_aligners,
    build_window_slices,
    compute_window_oracle_actions,
    load_ksda_v3_folds,
    oracle_usage_stats,
    paired_wins,
)
from src.alignment.koopman_alignment import KoopmanFeatureProjector  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d1")


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


def _evaluate_fold(fold, rep_cfg: Dict[str, object], window_size: int, lda_kwargs: Dict[str, object], cov_eps: float) -> Dict[str, object]:
    projector = KoopmanFeatureProjector(
        pca_rank=int(rep_cfg["pca_rank"]),
        lifting=str(rep_cfg["lifting"]),
    ).fit(fold.cov_source)
    psi_source = projector.transform(fold.cov_source)
    psi_target_train = projector.transform(fold.cov_target_train)
    psi_target_test = projector.transform(fold.cov_target_test)

    experts = build_local_expert_aligners(psi_source, psi_target_train, fold.y_source, cov_eps=cov_eps)
    expert_names = list(experts.keys())

    expert_predictions = {}
    expert_subject_rows = []
    window_scores = []
    oracle_labels = []
    oracle_predictions = np.zeros_like(fold.y_target_test, dtype=np.int64)
    for expert_name in expert_names:
        aligner = experts[expert_name]
        psi_source_aligned = apply_expert_aligner(aligner, psi_source)
        psi_target_aligned = apply_expert_aligner(aligner, psi_target_test)
        lda = LDA(**lda_kwargs).fit(psi_source_aligned, fold.y_source)
        y_pred = lda.predict(psi_target_aligned).astype(np.int64)
        expert_predictions[expert_name] = y_pred
        expert_subject_rows.append(
            {
                "target_subject": int(fold.target_subject),
                "expert": expert_name,
                **compute_metrics(fold.y_target_test, y_pred),
            }
        )

    switch_rows = []
    for window_idx, (start, end) in enumerate(build_window_slices(len(fold.y_target_test), int(window_size))):
        y_window = fold.y_target_test[start:end]
        accs = []
        for expert_name in expert_names:
            accs.append(float(np.mean(expert_predictions[expert_name][start:end] == y_window)))
        accs = np.asarray(accs, dtype=np.float64)
        chosen, _ = compute_window_oracle_actions(accs[None, :])
        chosen_idx = int(chosen[0])
        chosen_name = expert_names[chosen_idx]
        oracle_labels.append(chosen_name)
        oracle_predictions[start:end] = expert_predictions[chosen_name][start:end]
        switch_rows.append(
            {
                "window_id": int(window_idx),
                "oracle_expert": chosen_name,
                **{f"acc_{expert_name}": float(score) for expert_name, score in zip(expert_names, accs)},
            }
        )

    oracle_metrics = compute_metrics(fold.y_target_test, oracle_predictions)
    oracle_label_array = np.asarray(oracle_labels, dtype=object)
    transitions = int(np.sum(oracle_label_array[1:] != oracle_label_array[:-1])) if len(oracle_label_array) > 1 else 0
    usage = oracle_usage_stats(oracle_label_array)
    switch_summary = {
        "target_subject": int(fold.target_subject),
        "unique_experts": int(np.unique(oracle_label_array).size),
        "transitions": transitions,
        "most_common_ratio": float(usage["most_common_ratio"]),
    }

    return {
        "target_subject": int(fold.target_subject),
        "expert_subject_rows": expert_subject_rows,
        "oracle_metrics": {"target_subject": int(fold.target_subject), **oracle_metrics},
        "switch_summary": switch_summary,
        "oracle_labels": oracle_label_array,
        "window_rows": switch_rows,
        "expert_names": expert_names,
        "y_true": np.asarray(fold.y_target_test, dtype=np.int64),
        "trial_index": np.arange(len(fold.y_target_test), dtype=np.int64),
    }


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d0-run-dir", default=None)
    parser.add_argument("--d1r-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    d0_run_dir = _resolve_run_dir(args.d0_run_dir, "results/ksda/exp_d0")
    d0_summary = json.loads((d0_run_dir / "summary.json").read_text(encoding="utf-8"))
    if (not bool(d0_summary["dynamic_need_exists"])) and (not args.force):
        raise RuntimeError("D0 did not pass; refusing to run D1 without --force.")

    d1r_run_dir = Path(args.d1r_run_dir) if args.d1r_run_dir else Path(d0_summary["d1r_run_dir"])
    d1r_best = json.loads((d1r_run_dir / "best_static.json").read_text(encoding="utf-8"))
    rep_cfg = d1r_best["representation"]
    window_size = int(d0_summary["best_window_size"])

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

    root_dir = ensure_dir(f"results/ksda/exp_d1/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)

    expert_rows = []
    oracle_rows = []
    switch_rows = []
    oracle_hist = []
    expert_names = None
    for fold in folds:
        result = _evaluate_fold(fold, rep_cfg, window_size, lda_kwargs, cov_eps)
        expert_names = result["expert_names"]
        expert_rows.extend(result["expert_subject_rows"])
        oracle_rows.append(result["oracle_metrics"])
        switch_rows.append(result["switch_summary"])
        label_array = result["oracle_labels"]
        for expert_name in result["expert_names"]:
            oracle_hist.append(
                {
                    "target_subject": int(fold.target_subject),
                    "expert": expert_name,
                    "count": int(np.sum(label_array == expert_name)),
                    "ratio": float(np.mean(label_array == expert_name)),
                }
            )

        detail_df = pd.DataFrame(result["window_rows"])
        save_dict = {
            "y_true": result["y_true"],
            "trial_index": result["trial_index"],
            "window_id": detail_df["window_id"].to_numpy(dtype=np.int64),
        }
        save_dict["oracle_expert"] = detail_df["oracle_expert"].to_numpy(dtype=object)
        for expert_name in result["expert_names"]:
            save_dict[f"acc_{expert_name}"] = detail_df[f"acc_{expert_name}"].to_numpy(dtype=np.float64)
        np.savez(details_dir / f"subject_A{int(fold.target_subject):02d}.npz", **save_dict)

    elapsed = float(time.perf_counter() - start)
    expert_df = pd.DataFrame(expert_rows)
    oracle_df = pd.DataFrame(oracle_rows).sort_values("target_subject").reset_index(drop=True)
    switch_df = pd.DataFrame(switch_rows).sort_values("target_subject").reset_index(drop=True)
    hist_df = pd.DataFrame(oracle_hist)

    single_summary = (
        expert_df.groupby("expert")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
        .sort_values(["accuracy_mean", "expert"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_single_expert = str(single_summary.iloc[0]["expert"])
    best_single_subject = expert_df.loc[expert_df["expert"] == best_single_expert].sort_values("target_subject")
    pair = paired_wins(oracle_df["accuracy"], best_single_subject["accuracy"])

    hist_agg = hist_df.groupby("expert")["count"].sum().reset_index()
    total = max(int(hist_agg["count"].sum()), 1)
    hist_agg["ratio"] = hist_agg["count"] / total
    usage = oracle_usage_stats(np.repeat(hist_agg["expert"].to_numpy(dtype=object), hist_agg["count"].to_numpy(dtype=np.int64)))
    subjects_multi = int(np.sum(switch_df["unique_experts"].to_numpy(dtype=np.int64) >= 2))
    delta_subject = oracle_df["accuracy"].to_numpy(dtype=np.float64) - best_single_subject["accuracy"].to_numpy(dtype=np.float64)
    positive = np.clip(delta_subject, 0.0, None)
    max_contrib = float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0
    passed = bool(
        (
            pair["mean_delta"] >= 0.01
            or (pair["mean_delta"] >= 0.005 and pair["wins"] >= 6)
        )
        and usage["most_common_ratio"] < 0.8
        and subjects_multi >= 6
        and max_contrib <= 0.4
    )

    single_summary.to_csv(root_dir / "single_expert_summary.csv", index=False)
    oracle_summary = pd.DataFrame(
        [
            {
                "window_size": window_size,
                "oracle_accuracy_mean": float(oracle_df["accuracy"].mean()),
                "best_single_expert": best_single_expert,
                "best_single_accuracy_mean": float(best_single_subject["accuracy"].mean()),
                "mean_delta_vs_best_single": float(pair["mean_delta"]),
                "wins_vs_best_single": int(pair["wins"]),
                "losses_vs_best_single": int(pair["losses"]),
                "draws_vs_best_single": int(pair["draws"]),
                "most_common_ratio": float(usage["most_common_ratio"]),
                "subjects_with_multiple_experts": subjects_multi,
                "max_single_subject_contribution": max_contrib,
                "passed": passed,
            }
        ]
    )
    oracle_summary.to_csv(root_dir / "oracle_expert_summary.csv", index=False)
    hist_agg.to_csv(root_dir / "expert_action_histogram.csv", index=False)
    switch_df.to_csv(root_dir / "expert_switch_stats.csv", index=False)
    expert_df.to_csv(root_dir / "single_expert_loso.csv", index=False)
    oracle_df.to_csv(root_dir / "oracle_expert_loso.csv", index=False)

    summary = {
        "d0_run_dir": str(d0_run_dir),
        "d1r_run_dir": str(d1r_run_dir),
        "representation": rep_cfg,
        "window_size": window_size,
        "best_single_expert": best_single_expert,
        "dynamic_expert_need_exists": passed,
        "oracle_expert_summary": oracle_summary.iloc[0].to_dict(),
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plt.bar(single_summary["expert"], single_summary["accuracy_mean"])
    plt.axhline(float(oracle_df["accuracy"].mean()), linestyle="--", color="black", alpha=0.5, label="oracle")
    plt.ylabel("Accuracy")
    plt.title("D1 single experts vs oracle")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "single_experts_vs_oracle.pdf", dpi=300)
    plt.close()

    logger.info("Saved D1 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
