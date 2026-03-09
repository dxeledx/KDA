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
from src.evaluation.ksda_v31 import (  # noqa: E402
    compute_window_oracle_for_actions,
    load_trial_safe_fold_state,
)
from src.evaluation.ksda_v3 import build_window_slices, load_ksda_v3_folds  # noqa: E402
from src.evaluation.kcar_analysis import compute_transition_residuals, fit_koopman_operator, fit_subjectwise_global_koopman  # noqa: E402
from src.evaluation.rbid import summarize_local_k_rbid  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_k_rbid")
WINDOW_SIZE = 16


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
    args = parser.parse_args()

    d1t_run_dir = _resolve_run_dir(args.d1t_run_dir, "results/ksda/exp_d1t")
    d1t_summary = json.loads((d1t_run_dir / "summary.json").read_text(encoding="utf-8"))
    d1r_run_dir = Path(d1t_summary["d1r_run_dir"])
    d1r_best = json.loads((d1r_run_dir / "best_static.json").read_text(encoding="utf-8"))
    rep_cfg = d1t_summary["representation"]
    best_fixed_action = str(d1t_summary["best_single_action"])

    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))
    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    all_subjects = list(map(int, data_cfg["dataset"]["subjects"]))
    target_subjects = _resolve_targets(all_subjects, args.targets)
    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/ksda/k_rbid/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)
    rows = []
    for fold in folds:
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold.cov_source)
        state = load_trial_safe_fold_state(fold, projector, LDA, model_cfg["lda"], cov_eps=cov_eps)
        action_predictions = {
            name: np.asarray(payload["y_pred"], dtype=np.int64)
            for name, payload in state["action_results"].items()
        }
        action_names = list(action_predictions.keys())

        source_blocks = []
        z_source = projector.transform_tangent(fold.cov_source)
        start_idx = 0
        for length in fold.source_block_lengths:
            source_blocks.append(z_source[start_idx : start_idx + int(length)])
            start_idx += int(length)
        oracle = compute_window_oracle_for_actions(
            np.asarray(fold.y_target_test, dtype=np.int64),
            action_predictions,
            window_size=WINDOW_SIZE,
        )
        for window_idx, (start_w, end_w) in enumerate(build_window_slices(len(fold.y_target_test), WINDOW_SIZE)):
            y_window = fold.y_target_test[start_w:end_w]
            geometry_scores = []
            behavior_scores = []
            for action_name in action_names:
                pred_window = action_predictions[action_name][start_w:end_w]
                behavior_scores.append(float(np.mean(pred_window == y_window) - np.mean(action_predictions[best_fixed_action][start_w:end_w] == y_window)))
                source_transformed = np.asarray(state["action_results"][action_name]["source_transformed"], dtype=np.float64)
                target_train_transformed = np.asarray(state["action_results"][action_name]["target_train_transformed"], dtype=np.float64)
                target_test_transformed = np.asarray(state["action_results"][action_name]["target_transformed"], dtype=np.float64)
                source_operator = fit_koopman_operator(source_transformed, ridge_alpha=1.0e-3)
                target_operator = fit_koopman_operator(target_train_transformed, ridge_alpha=1.0e-3)
                target_window = target_test_transformed[start_w:end_w]
                if len(target_window) >= 2:
                    e_src = compute_transition_residuals(target_window, source_operator)
                    e_tgt = compute_transition_residuals(target_window, target_operator)
                    geometry_scores.append(-float(np.mean(e_tgt - e_src)))
                else:
                    geometry_scores.append(0.0)
            metrics = summarize_local_k_rbid(
                np.asarray([geometry_scores], dtype=np.float64),
                np.asarray([behavior_scores], dtype=np.float64),
            )
            rows.append(
                {
                    "target_subject": int(fold.target_subject),
                    "window_id": int(window_idx),
                    "oracle_action": action_names[int(oracle["oracle_action_indices"][window_idx])],
                    "k_rbid": float(metrics["k_rbid_per_window"][0]),
                    "k_rbid_pos": float(metrics["k_rbid_pos_per_window"][0]),
                    "k_rbid_neg": float(metrics["k_rbid_neg_per_window"][0]),
                    "oracle_gain": float(np.max(behavior_scores)),
                }
            )

    elapsed = float(time.perf_counter() - start)
    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "local_k_rbid_window_metrics.csv", index=False)
    corr = float(df[["k_rbid", "oracle_gain"]].corr().iloc[0, 1]) if len(df) > 1 else 0.0
    grouped = df.groupby(pd.qcut(df["k_rbid"], q=2, duplicates="drop"))["oracle_gain"].mean().reset_index(name="mean_oracle_gain")
    grouped.to_csv(root_dir / "k_rbid_vs_gain.csv", index=False)
    summary = {
        "d1t_run_dir": str(d1t_run_dir),
        "window_size": WINDOW_SIZE,
        "corr_k_rbid_vs_oracle_gain": corr,
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "k_rbid_diagnostics_summary.json")
    (root_dir / "k_rbid_examples.md").write_text(
        df.sort_values("k_rbid", ascending=False).head(10).to_markdown(index=False),
        encoding="utf-8",
    )

    plt.figure(figsize=(7.5, 4.5))
    plt.scatter(df["k_rbid"], df["oracle_gain"], alpha=0.7)
    plt.xlabel("K-RBID_t")
    plt.ylabel("Pseudo-oracle gain")
    plt.title("K-RBID vs window gain")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "k_rbid_vs_gain.pdf", dpi=300)
    plt.close()

    logger.info("Saved K-RBID diagnostics to %s", root_dir)


if __name__ == "__main__":
    main()
