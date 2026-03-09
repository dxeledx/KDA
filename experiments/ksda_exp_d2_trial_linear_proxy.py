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
    build_causal_trialized_teacher_from_state,
    compute_trial_features,
    fit_linear_multiclass_selector,
    load_custom_ksda_fold,
    load_trial_safe_fold_state,
    paired_wins,
    predict_linear_multiclass_selector,
)
from src.evaluation.ksda_v3 import load_ksda_v3_folds  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d2_trial")
TRAILING_LEN = 16
TEACHER_WINDOW = 16
GEOM_ONLY_INDICES = [3, 4, 5]


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


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d1p5-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    d1p5_run_dir = _resolve_run_dir(args.d1p5_run_dir, "results/ksda/exp_d1p5")
    d1p5_summary = json.loads((d1p5_run_dir / "summary.json").read_text(encoding="utf-8"))
    if (not bool(d1p5_summary["causal_trialization_valid"])) and (not args.force):
        raise RuntimeError("D1.5 did not pass; refusing to run D2 without --force.")

    d1t_run_dir = Path(d1p5_summary["d1t_run_dir"])
    d1r_run_dir = Path(json.loads((d1t_run_dir / "summary.json").read_text(encoding="utf-8"))["d1r_run_dir"])
    d1r_best = json.loads((d1r_run_dir / "best_static.json").read_text(encoding="utf-8"))
    rep_cfg = d1r_best["representation"]
    best_single_action = str(json.loads((d1t_run_dir / "summary.json").read_text(encoding="utf-8"))["best_single_action"])

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

    root_dir = ensure_dir(f"results/ksda/exp_d2_trial/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")
    models_dir = ensure_dir(root_dir / "models")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)

    rows = []
    agreement_rows = []
    for fold in folds:
        target = int(fold.target_subject)
        source_subjects = [subject for subject in all_subjects if subject != target]

        X_train_blocks = []
        y_train_blocks = []
        for pseudo_target in source_subjects:
            pseudo_sources = [subject for subject in source_subjects if subject != pseudo_target]
            pseudo_fold = load_custom_ksda_fold(loader, pseudo_sources, pseudo_target, pre, cov_eps)
            projector = KoopmanFeatureProjector(
                pca_rank=int(rep_cfg["pca_rank"]),
                lifting=str(rep_cfg["lifting"]),
            ).fit(pseudo_fold.cov_source)
            pseudo_state = load_trial_safe_fold_state(pseudo_fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)
            teacher = build_causal_trialized_teacher_from_state(
                pseudo_fold,
                pseudo_state,
                window_size=TEACHER_WINDOW,
            )
            features = compute_trial_features(
                pseudo_state["z_target_test"],
                pseudo_state["psi_target_test"],
                pseudo_state["source_operator"],
                pseudo_state["target_operator"],
                pseudo_state["raw_lda"],
                trailing_len=TRAILING_LEN,
                source_mean_z=pseudo_state["source_mean_z"],
                source_diag_var_z=pseudo_state["source_diag_var_z"],
            )
            X_train_blocks.append(features)
            y_train_blocks.append(np.asarray(teacher["teacher_trial_indices"], dtype=np.int64))

        X_train = np.concatenate(X_train_blocks, axis=0)
        y_train = np.concatenate(y_train_blocks, axis=0)
        selector = fit_linear_multiclass_selector(X_train, y_train, num_classes=13, reg_lambda=1.0e-3)
        geom_selector = fit_linear_multiclass_selector(X_train[:, GEOM_ONLY_INDICES], y_train, num_classes=13, reg_lambda=1.0e-3)

        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold.cov_source)
        state = load_trial_safe_fold_state(fold, projector, LDA, lda_kwargs, cov_eps=cov_eps)
        teacher = build_causal_trialized_teacher_from_state(
            fold,
            state,
            window_size=TEACHER_WINDOW,
        )
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
        chosen_full, scores_full = predict_linear_multiclass_selector(selector, features)
        chosen_geom, scores_geom = predict_linear_multiclass_selector(geom_selector, features[:, GEOM_ONLY_INDICES])
        teacher_labels = np.asarray(teacher["teacher_trial_indices"], dtype=np.int64)

        y_pred_full = _apply_trial_action_selector(
            {name: np.asarray(payload["y_pred"], dtype=np.int64) for name, payload in state["action_results"].items()},
            state["action_names"],
            chosen_full,
        )
        y_pred_geom = _apply_trial_action_selector(
            {name: np.asarray(payload["y_pred"], dtype=np.int64) for name, payload in state["action_results"].items()},
            state["action_names"],
            chosen_geom,
        )
        y_pred_best = np.asarray(state["action_results"][best_single_action]["y_pred"], dtype=np.int64)
        y_pred_teacher = np.asarray(teacher["teacher_pred"], dtype=np.int64)

        rows.extend(
            [
                {"method": "best_single_action", "target_subject": target, **compute_metrics(fold.y_target_test, y_pred_best)},
                {"method": "causal_trialized_oracle", "target_subject": target, **compute_metrics(fold.y_target_test, y_pred_teacher)},
                {"method": "linear_proxy_selector", "target_subject": target, **compute_metrics(fold.y_target_test, y_pred_full)},
                {"method": "geom_only_selector", "target_subject": target, **compute_metrics(fold.y_target_test, y_pred_geom)},
            ]
        )
        agreement_rows.append(
            {
                "target_subject": target,
                "agreement_to_teacher_full": float(np.mean(chosen_full == teacher_labels)),
                "agreement_to_teacher_geom": float(np.mean(chosen_geom == teacher_labels)),
            }
        )
        np.savez(
            details_dir / f"subject_A{target:02d}.npz",
            trial_index=np.arange(len(fold.y_target_test), dtype=np.int64),
            y_true=np.asarray(fold.y_target_test, dtype=np.int64),
            teacher_action_t=teacher_labels,
            chosen_full=chosen_full,
            chosen_geom=chosen_geom,
            y_pred_full=y_pred_full,
            y_pred_geom=y_pred_geom,
            y_pred_teacher=y_pred_teacher,
            y_pred_best=y_pred_best,
            features=features,
            scores_full=scores_full,
            scores_geom=scores_geom,
            action_names=np.asarray(state["action_names"], dtype=object),
        )
        np.savez(
            models_dir / f"target_A{target:02d}.npz",
            mean=selector["mean"],
            std=selector["std"],
            weights=selector["weights"],
            geom_mean=geom_selector["mean"],
            geom_std=geom_selector["std"],
            geom_weights=geom_selector["weights"],
            action_names=np.asarray(state["action_names"], dtype=object),
        )

    elapsed = float(time.perf_counter() - start)
    rows_df = pd.DataFrame(rows)
    agreement_df = pd.DataFrame(agreement_rows).sort_values("target_subject").reset_index(drop=True)
    summary_df = (
        rows_df.groupby("method")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    )

    best_df = rows_df.loc[rows_df["method"] == "best_single_action"].sort_values("target_subject")
    teacher_df = rows_df.loc[rows_df["method"] == "causal_trialized_oracle"].sort_values("target_subject")
    full_df = rows_df.loc[rows_df["method"] == "linear_proxy_selector"].sort_values("target_subject")
    geom_df = rows_df.loc[rows_df["method"] == "geom_only_selector"].sort_values("target_subject")

    full_vs_geom = paired_wins(full_df["accuracy"], geom_df["accuracy"])
    full_vs_best = paired_wins(full_df["accuracy"], best_df["accuracy"])
    agreement_full = float(agreement_df["agreement_to_teacher_full"].mean())
    random_baseline = 1.0 / 13.0
    passed = bool(
        full_vs_geom["mean_delta"] >= 0.005
        and full_vs_best["mean_delta"] >= 0.0
        and agreement_full >= 0.45
        and agreement_full >= random_baseline + 0.15
    )

    rows_df.to_csv(root_dir / "selector_loso.csv", index=False)
    agreement_df.to_csv(root_dir / "agreement_summary.csv", index=False)
    summary_df.to_csv(root_dir / "selector_summary.csv", index=False)
    summary = {
        "d1p5_run_dir": str(d1p5_run_dir),
        "representation": rep_cfg,
        "best_single_action": best_single_action,
        "pairwise": {
            "linear_vs_geom": full_vs_geom,
            "linear_vs_best_single": full_vs_best,
        },
        "oracle_agreement_full": agreement_full,
        "oracle_agreement_geom": float(agreement_df["agreement_to_teacher_geom"].mean()),
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plot_df = summary_df.set_index("method").loc[
        ["best_single_action", "geom_only_selector", "linear_proxy_selector", "causal_trialized_oracle"]
    ].reset_index()
    plt.bar(plot_df["method"], plot_df["accuracy_mean"])
    plt.ylabel("Accuracy")
    plt.title("D2 trial-level selector comparison")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(root_dir / "figures" / "selector_comparison.pdf", dpi=300)
    plt.close()

    logger.info("Saved D2 trial outputs to %s", root_dir)


if __name__ == "__main__":
    main()
