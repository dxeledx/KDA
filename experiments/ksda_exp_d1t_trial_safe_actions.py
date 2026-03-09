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
    load_trial_safe_fold_state,
    resolve_representation_config,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d1t")


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
    parser.add_argument("--d1r-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--pca-rank", type=int, default=None)
    parser.add_argument("--lifting", default=None)
    args = parser.parse_args()

    d1r_run_dir = _resolve_run_dir(args.d1r_run_dir, "results/ksda/exp_d1r")
    d1r_best = json.loads((d1r_run_dir / "best_static.json").read_text(encoding="utf-8"))
    rep_cfg = resolve_representation_config(
        d1r_best["representation"],
        pca_rank=args.pca_rank,
        lifting=args.lifting,
    )

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

    root_dir = ensure_dir(f"results/ksda/exp_d1t/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)

    single_rows = []
    overlap_rows = []
    delta_rows = []
    summary_rows = []
    for fold in folds:
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold.cov_source)
        state = load_trial_safe_fold_state(fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)

        identity_pred = np.asarray(state["action_results"]["A0_identity"]["y_pred"], dtype=np.int64)
        raw_norm = np.linalg.norm(state["psi_target_test"], axis=1)
        save_dict = {
            "y_true": np.asarray(fold.y_target_test, dtype=np.int64),
            "trial_index": np.arange(len(fold.y_target_test), dtype=np.int64),
            "raw_feature_norm": raw_norm.astype(np.float64),
        }

        for action_name, payload in state["action_results"].items():
            y_pred = np.asarray(payload["y_pred"], dtype=np.int64)
            transform_delta = np.asarray(payload["transform_delta"], dtype=np.float64)
            metrics = compute_metrics(fold.y_target_test, y_pred)
            single_rows.append(
                {"target_subject": int(fold.target_subject), "action": action_name, **metrics}
            )
            if action_name != "A0_identity":
                overlap_rows.append(
                    {
                        "target_subject": int(fold.target_subject),
                        "action": action_name,
                        "overlap_vs_identity": float(np.mean(y_pred == identity_pred)),
                    }
                )
                delta_rows.append(
                    {
                        "target_subject": int(fold.target_subject),
                        "action": action_name,
                        "mean_transform_delta": float(np.mean(transform_delta)),
                        "relative_transform_delta": float(np.mean(transform_delta) / max(float(np.mean(raw_norm)), 1.0e-8)),
                    }
                )
            save_dict[f"pred_{action_name}"] = y_pred
            save_dict[f"delta_{action_name}"] = transform_delta

        summary_rows.append({"target_subject": int(fold.target_subject), **state["summary"]})
        np.savez(details_dir / f"subject_A{int(fold.target_subject):02d}.npz", **save_dict)

    elapsed = float(time.perf_counter() - start)
    single_df = pd.DataFrame(single_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    delta_df = pd.DataFrame(delta_rows)
    subject_summary_df = pd.DataFrame(summary_rows).sort_values("target_subject").reset_index(drop=True)

    action_summary = (
        single_df.groupby("action")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
        .sort_values(["accuracy_mean", "action"], ascending=[False, True])
        .reset_index(drop=True)
    )
    overlap_summary = (
        overlap_df.groupby("action")["overlap_vs_identity"]
        .mean()
        .reset_index()
        .sort_values(["overlap_vs_identity", "action"], ascending=[False, True])
        .reset_index(drop=True)
    )
    delta_summary = (
        delta_df.groupby("action")[["mean_transform_delta", "relative_transform_delta"]]
        .mean()
        .reset_index()
        .sort_values(["relative_transform_delta", "action"], ascending=[True, True])
        .reset_index(drop=True)
    )

    top3_families = [name.split("_")[0] for name in action_summary["action"].head(3).tolist()]
    high_overlap_actions = overlap_summary.loc[overlap_summary["overlap_vs_identity"] > 0.95, "action"].tolist()
    low_delta_actions = delta_summary.loc[delta_summary["relative_transform_delta"] < 0.01, "action"].tolist()
    passed = not (len(high_overlap_actions) >= 2 or len(low_delta_actions) >= 2 or len(set(top3_families)) == 1)

    action_summary.to_csv(root_dir / "single_action_trial_summary.csv", index=False)
    overlap_summary.to_csv(root_dir / "action_overlap_vs_identity.csv", index=False)
    delta_summary.to_csv(root_dir / "action_transform_delta.csv", index=False)
    single_df.to_csv(root_dir / "single_action_loso.csv", index=False)
    subject_summary_df.to_csv(root_dir / "subject_trial_safe_summary.csv", index=False)

    summary = {
        "d1r_run_dir": str(d1r_run_dir),
        "representation": rep_cfg,
        "best_single_action": str(action_summary.iloc[0]["action"]),
        "best_single_accuracy_mean": float(action_summary.iloc[0]["accuracy_mean"]),
        "high_overlap_actions": high_overlap_actions,
        "low_delta_actions": low_delta_actions,
        "top3_action_families": top3_families,
        "trial_safe_action_space_valid": passed,
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(10, 4.8))
    plt.bar(action_summary["action"], action_summary["accuracy_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("D1-T single action benchmark")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "single_action_accuracy.pdf", dpi=300)
    plt.close()

    logger.info("Saved D1-T outputs to %s", root_dir)


if __name__ == "__main__":
    main()
