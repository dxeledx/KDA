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
    summarize_rank_scan_metrics,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d1t0")
DEFAULT_RANKS = [8, 16, 24, 32, 48, 64]


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


def _parse_ranks(raw: str | None) -> List[int]:
    if not raw:
        return list(DEFAULT_RANKS)
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d1r-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--pca-ranks", default="8,16,24,32,48,64")
    parser.add_argument("--lifting", default="quadratic")
    args = parser.parse_args()

    d1r_run_dir = _resolve_run_dir(args.d1r_run_dir, "results/ksda/exp_d1r")

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
    pca_ranks = _parse_ranks(args.pca_ranks)
    lifting = str(args.lifting)

    root_dir = ensure_dir(f"results/ksda/exp_d1t0/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)

    rank_rows = []
    per_action_rows = []
    per_overlap_rows = []
    for rank in pca_ranks:
        subject_summaries = []
        single_rows = []
        overlap_rows = []
        for fold in folds:
            projector = KoopmanFeatureProjector(
                pca_rank=int(rank),
                lifting=lifting,
            ).fit(fold.cov_source)
            state = load_trial_safe_fold_state(fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)
            summary = state["summary"]
            subject_summaries.append(summary)
            identity_pred = np.asarray(state["action_results"]["A0_identity"]["y_pred"], dtype=np.int64)
            for action_name, payload in state["action_results"].items():
                y_pred = np.asarray(payload["y_pred"], dtype=np.int64)
                metrics = compute_metrics(fold.y_target_test, y_pred)
                single_rows.append(
                    {
                        "pca_rank": int(rank),
                        "target_subject": int(fold.target_subject),
                        "action": action_name,
                        **metrics,
                    }
                )
                if action_name != "A0_identity":
                    overlap_rows.append(
                        {
                            "pca_rank": int(rank),
                            "target_subject": int(fold.target_subject),
                            "action": action_name,
                            "overlap_vs_identity": float(np.mean(y_pred == identity_pred)),
                        }
                    )

            save_dict = {
                "trial_index": np.arange(len(fold.y_target_test), dtype=np.int64),
                "y_true": np.asarray(fold.y_target_test, dtype=np.int64),
            }
            for action_name, payload in state["action_results"].items():
                save_dict[f"pred_{action_name}"] = np.asarray(payload["y_pred"], dtype=np.int64)
            np.savez(details_dir / f"rank{rank}_subject_A{int(fold.target_subject):02d}.npz", **save_dict)

        single_df = pd.DataFrame(single_rows)
        overlap_df = pd.DataFrame(overlap_rows)
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
        overlap_map = dict(zip(overlap_summary["action"], overlap_summary["overlap_vs_identity"]))
        rank_summary = summarize_rank_scan_metrics(
            best_single_action=str(action_summary.iloc[0]["action"]),
            best_single_accuracy=float(action_summary.iloc[0]["accuracy_mean"]),
            overlap_vs_identity=overlap_map,
        )
        rank_summary["pca_rank"] = int(rank)
        rank_summary["lifting"] = lifting
        rank_rows.append(rank_summary)

        per_action_rows.extend(action_summary.assign(pca_rank=int(rank), lifting=lifting).to_dict(orient="records"))
        per_overlap_rows.extend(overlap_summary.assign(pca_rank=int(rank), lifting=lifting).to_dict(orient="records"))
        logger.info(
            "D1-T.0 rank=%s best=%s acc=%.4f high_overlap=%s",
            rank,
            rank_summary["best_single_action"],
            rank_summary["best_single_accuracy"],
            rank_summary["num_high_overlap_actions"],
        )

    elapsed = float(time.perf_counter() - start)
    rank_df = pd.DataFrame(rank_rows).sort_values(["best_single_accuracy", "max_overlap_vs_identity"], ascending=[False, True])
    per_action_df = pd.DataFrame(per_action_rows)
    per_overlap_df = pd.DataFrame(per_overlap_rows)

    rank_df.to_csv(root_dir / "rank_scan_summary.csv", index=False)
    per_action_df.to_csv(root_dir / "per_rank_action_accuracy.csv", index=False)
    per_overlap_df.to_csv(root_dir / "per_rank_action_overlap.csv", index=False)
    summary = {
        "d1r_run_dir": str(d1r_run_dir),
        "lifting": lifting,
        "pca_ranks": pca_ranks,
        "best_by_accuracy": rank_df.sort_values(["best_single_accuracy", "max_overlap_vs_identity"], ascending=[False, True]).iloc[0].to_dict(),
        "best_by_decrowding": rank_df.sort_values(["num_high_overlap_actions", "max_overlap_vs_identity", "best_single_accuracy"], ascending=[True, True, False]).iloc[0].to_dict(),
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plt.plot(rank_df["pca_rank"], rank_df["best_single_accuracy"], marker="o", label="best action acc")
    plt.xlabel("PCA rank")
    plt.ylabel("Accuracy")
    plt.title("D1-T.0 best single action accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "rank_vs_accuracy.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(8.0, 4.5))
    plt.plot(rank_df["pca_rank"], rank_df["num_high_overlap_actions"], marker="o", color="tab:red", label="#high-overlap actions")
    plt.xlabel("PCA rank")
    plt.ylabel("Count")
    plt.title("D1-T.0 action de-crowding diagnostic")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "rank_vs_overlap_count.pdf", dpi=300)
    plt.close()

    logger.info("Saved D1-T.0 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
