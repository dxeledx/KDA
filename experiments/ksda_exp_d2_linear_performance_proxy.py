#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.koopman_alignment import KoopmanFeatureProjector  # noqa: E402
from src.evaluation.kcar_analysis import (  # noqa: E402
    fit_koopman_operator,
    fit_subjectwise_global_koopman,
)
from src.evaluation.ksda_v3 import (  # noqa: E402
    KSDAV3Fold,
    apply_expert_aligner,
    build_local_expert_aligners,
    build_window_slices,
    compute_window_feature_matrix,
    compute_window_oracle_actions,
    fit_linear_multiclass_selector,
    fit_linear_scalar_proxy,
    load_ksda_v3_folds,
    paired_wins,
    predict_linear_multiclass_selector,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d2")

GEOM_FEATURE_INDICES = [3, 4, 5]


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


def _resolve_targets(all_subjects: Sequence[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def _load_custom_fold(loader, source_subjects: Sequence[int], target_subject: int, pre, cov_eps: float) -> KSDAV3Fold:
    X_source_blocks, y_source_blocks = [], []
    for subject in source_subjects:
        X_train_subject, y_train_subject = loader.load_subject(int(subject), split="train")
        X_source_blocks.append(X_train_subject)
        y_source_blocks.append(y_train_subject)

    X_source_raw = np.concatenate(X_source_blocks, axis=0)
    y_source = np.concatenate(y_source_blocks, axis=0)
    X_target_train_raw, _ = loader.load_subject(int(target_subject), split="train")
    X_target_test_raw, y_target_test = loader.load_subject(int(target_subject), split="test")

    X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
    X_target_train = pre.transform(X_target_train_raw)
    X_target_test = pre.transform(X_target_test_raw)

    from src.features.covariance import compute_covariances

    return KSDAV3Fold(
        target_subject=int(target_subject),
        cov_source=compute_covariances(X_source, eps=cov_eps),
        y_source=np.asarray(y_source, dtype=np.int64),
        cov_target_train=compute_covariances(X_target_train, eps=cov_eps),
        cov_target_test=compute_covariances(X_target_test, eps=cov_eps),
        y_target_test=np.asarray(y_target_test, dtype=np.int64),
        source_block_lengths=[block.shape[0] for block in X_source_blocks],
    )


def _build_fold_state(fold: KSDAV3Fold, rep_cfg: Dict[str, object], lda_kwargs: Dict[str, object], cov_eps: float) -> Dict[str, object]:
    projector = KoopmanFeatureProjector(
        pca_rank=int(rep_cfg["pca_rank"]),
        lifting=str(rep_cfg["lifting"]),
    ).fit(fold.cov_source)
    psi_source = projector.transform(fold.cov_source)
    psi_target_train = projector.transform(fold.cov_target_train)
    psi_target_test = projector.transform(fold.cov_target_test)
    z_source = projector.transform_tangent(fold.cov_source)
    z_target_train = projector.transform_tangent(fold.cov_target_train)
    z_target_test = projector.transform_tangent(fold.cov_target_test)

    experts = build_local_expert_aligners(psi_source, psi_target_train, fold.y_source, cov_eps=cov_eps)
    expert_names = list(experts.keys())
    expert_predictions = {}
    expert_window_scores = {}
    global_accuracies = {}
    for expert_name, expert in experts.items():
        psi_source_aligned = apply_expert_aligner(expert, psi_source)
        psi_target_aligned = apply_expert_aligner(expert, psi_target_test)
        lda = LDA(**lda_kwargs).fit(psi_source_aligned, fold.y_source)
        pred = lda.predict(psi_target_aligned).astype(np.int64)
        expert_predictions[expert_name] = pred
        global_accuracies[expert_name] = float(np.mean(pred == fold.y_target_test))

    raw_lda = LDA(**lda_kwargs).fit(psi_source, fold.y_source)
    source_blocks = []
    start = 0
    for length in fold.source_block_lengths:
        source_blocks.append(z_source[start : start + int(length)])
        start += int(length)
    source_operator = fit_subjectwise_global_koopman(source_blocks, ridge_alpha=1.0e-3)
    target_operator = fit_koopman_operator(z_target_train, ridge_alpha=1.0e-3)
    source_mean_z = np.mean(np.concatenate(source_blocks, axis=0), axis=0)
    source_diag_var_z = np.var(np.concatenate(source_blocks, axis=0), axis=0)
    return {
        "projector": projector,
        "psi_source": psi_source,
        "psi_target_test": psi_target_test,
        "z_target_test": z_target_test,
        "expert_names": expert_names,
        "expert_predictions": expert_predictions,
        "global_accuracies": global_accuracies,
        "raw_lda": raw_lda,
        "source_operator": source_operator,
        "target_operator": target_operator,
        "source_mean_z": source_mean_z,
        "source_diag_var_z": source_diag_var_z,
    }


def _window_oracle_labels(fold: KSDAV3Fold, state: Dict[str, object], window_size: int) -> Dict[str, object]:
    expert_names = list(state["expert_names"])
    rows = []
    best_single_name = max(state["global_accuracies"].items(), key=lambda item: item[1])[0]
    gain_targets = []
    for start, end in build_window_slices(len(fold.y_target_test), window_size):
        y_window = fold.y_target_test[start:end]
        scores = []
        for expert_name in expert_names:
            scores.append(float(np.mean(state["expert_predictions"][expert_name][start:end] == y_window)))
        scores = np.asarray(scores, dtype=np.float64)
        chosen_idx, oracle_acc = compute_window_oracle_actions(scores[None, :])
        chosen_idx = int(chosen_idx[0])
        rows.append(
            {
                "label_idx": chosen_idx,
                "label_name": expert_names[chosen_idx],
                "oracle_acc": float(oracle_acc),
                "best_single_acc": float(scores[expert_names.index(best_single_name)]),
            }
        )
        gain_targets.append(float(oracle_acc - scores[expert_names.index(best_single_name)]))
    return {
        "rows": rows,
        "gain_targets": np.asarray(gain_targets, dtype=np.float64),
        "best_single_name": best_single_name,
    }


def _feature_matrix_for_fold(fold: KSDAV3Fold, state: Dict[str, object], window_size: int) -> Dict[str, np.ndarray]:
    return compute_window_feature_matrix(
        state["z_target_test"],
        state["psi_target_test"],
        state["source_operator"],
        state["target_operator"],
        state["raw_lda"],
        window_size,
        state["source_mean_z"],
        state["source_diag_var_z"],
    )


def _apply_window_selector(
    expert_predictions: Dict[str, np.ndarray],
    expert_names: List[str],
    chosen_labels: np.ndarray,
    total_trials: int,
    window_size: int,
) -> np.ndarray:
    y_pred = np.zeros(total_trials, dtype=np.int64)
    for window_idx, (start, end) in enumerate(build_window_slices(total_trials, window_size)):
        name = expert_names[int(chosen_labels[window_idx])]
        y_pred[start:end] = expert_predictions[name][start:end]
    return y_pred


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d1-run-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    d1_run_dir = _resolve_run_dir(args.d1_run_dir, "results/ksda/exp_d1")
    d1_summary = json.loads((d1_run_dir / "summary.json").read_text(encoding="utf-8"))
    if (not bool(d1_summary["dynamic_expert_need_exists"])) and (not args.force):
        raise RuntimeError("D1 did not pass; refusing to run D2 without --force.")

    rep_cfg = d1_summary["representation"]
    window_size = int(d1_summary["window_size"])
    best_single_expert = str(d1_summary["best_single_expert"])

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

    root_dir = ensure_dir(f"results/ksda/exp_d2/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")
    models_dir = ensure_dir(root_dir / "models")

    start = time.perf_counter()
    folds = load_ksda_v3_folds(loader, all_subjects, target_subjects, pre, cov_eps)

    summary_rows = []
    agreement_rows = []
    for fold in folds:
        target = int(fold.target_subject)
        source_subjects = [subject for subject in all_subjects if subject != target]

        X_train_rows, y_train_rows, gain_train_rows = [], [], []
        expert_names = None
        for pseudo_target in source_subjects:
            pseudo_sources = [subject for subject in source_subjects if subject != pseudo_target]
            pseudo_fold = _load_custom_fold(loader, pseudo_sources, pseudo_target, pre, cov_eps)
            pseudo_state = _build_fold_state(pseudo_fold, rep_cfg, lda_kwargs, cov_eps)
            expert_names = pseudo_state["expert_names"]
            feature_bundle = _feature_matrix_for_fold(pseudo_fold, pseudo_state, window_size)
            oracle = _window_oracle_labels(pseudo_fold, pseudo_state, window_size)
            X_train_rows.append(feature_bundle["features"])
            y_train_rows.append(np.asarray([row["label_idx"] for row in oracle["rows"]], dtype=np.int64))
            gain_train_rows.append(np.asarray(oracle["gain_targets"], dtype=np.float64))

        X_train = np.concatenate(X_train_rows, axis=0)
        y_train = np.concatenate(y_train_rows, axis=0)
        gain_train = np.concatenate(gain_train_rows, axis=0)
        selector_full = fit_linear_multiclass_selector(X_train, y_train, num_classes=len(expert_names))
        selector_geom = fit_linear_multiclass_selector(X_train[:, GEOM_FEATURE_INDICES], y_train, num_classes=len(expert_names))
        gain_proxy = fit_linear_scalar_proxy(X_train, gain_train)

        actual_state = _build_fold_state(fold, rep_cfg, lda_kwargs, cov_eps)
        feature_bundle = _feature_matrix_for_fold(fold, actual_state, window_size)
        oracle = _window_oracle_labels(fold, actual_state, window_size)
        oracle_labels = np.asarray([row["label_idx"] for row in oracle["rows"]], dtype=np.int64)

        chosen_full, scores_full = predict_linear_multiclass_selector(selector_full, feature_bundle["features"])
        chosen_geom, scores_geom = predict_linear_multiclass_selector(selector_geom, feature_bundle["features"][:, GEOM_FEATURE_INDICES])
        y_pred_full = _apply_window_selector(actual_state["expert_predictions"], actual_state["expert_names"], chosen_full, len(fold.y_target_test), window_size)
        y_pred_geom = _apply_window_selector(actual_state["expert_predictions"], actual_state["expert_names"], chosen_geom, len(fold.y_target_test), window_size)
        y_pred_best = actual_state["expert_predictions"][best_single_expert]
        y_pred_oracle = _apply_window_selector(actual_state["expert_predictions"], actual_state["expert_names"], oracle_labels, len(fold.y_target_test), window_size)

        full_metrics = compute_metrics(fold.y_target_test, y_pred_full)
        geom_metrics = compute_metrics(fold.y_target_test, y_pred_geom)
        best_metrics = compute_metrics(fold.y_target_test, y_pred_best)
        oracle_metrics = compute_metrics(fold.y_target_test, y_pred_oracle)

        summary_rows.extend(
            [
                {"method": "best_single_expert", "target_subject": target, **best_metrics},
                {"method": "oracle_expert", "target_subject": target, **oracle_metrics},
                {"method": "linear_proxy_selector", "target_subject": target, **full_metrics},
                {"method": "geom_only_selector", "target_subject": target, **geom_metrics},
            ]
        )
        agreement_rows.append(
            {
                "target_subject": target,
                "oracle_agreement_full": float(np.mean(chosen_full == oracle_labels)),
                "oracle_agreement_geom": float(np.mean(chosen_geom == oracle_labels)),
            }
        )
        np.savez(
            details_dir / f"subject_A{target:02d}.npz",
            y_true=np.asarray(fold.y_target_test, dtype=np.int64),
            oracle_labels=oracle_labels,
            chosen_full=chosen_full,
            chosen_geom=chosen_geom,
            expert_names=np.asarray(actual_state["expert_names"], dtype=object),
            features=feature_bundle["features"],
            trial_index=np.arange(len(fold.y_target_test), dtype=np.int64),
            window_id=feature_bundle["window_id"],
        )
        np.savez(
            models_dir / f"target_A{target:02d}.npz",
            selector_mean=selector_full["mean"],
            selector_std=selector_full["std"],
            selector_weights=selector_full["weights"],
            geom_mean=selector_geom["mean"],
            geom_std=selector_geom["std"],
            geom_weights=selector_geom["weights"],
            gain_mean=gain_proxy["mean"],
            gain_std=gain_proxy["std"],
            gain_beta=gain_proxy["beta"],
            expert_names=np.asarray(expert_names, dtype=object),
            window_size=np.asarray([window_size], dtype=np.int64),
        )

    elapsed = float(time.perf_counter() - start)
    summary_df = pd.DataFrame(summary_rows)
    agreement_df = pd.DataFrame(agreement_rows).sort_values("target_subject").reset_index(drop=True)
    method_summary = (
        summary_df.groupby("method")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    )

    best_df = summary_df.loc[summary_df["method"] == "best_single_expert"].sort_values("target_subject")
    oracle_df = summary_df.loc[summary_df["method"] == "oracle_expert"].sort_values("target_subject")
    full_df = summary_df.loc[summary_df["method"] == "linear_proxy_selector"].sort_values("target_subject")
    geom_df = summary_df.loc[summary_df["method"] == "geom_only_selector"].sort_values("target_subject")
    full_vs_geom = paired_wins(full_df["accuracy"], geom_df["accuracy"])
    full_vs_best = paired_wins(full_df["accuracy"], best_df["accuracy"])
    oracle_agreement = float(agreement_df["oracle_agreement_full"].mean())
    geom_agreement = float(agreement_df["oracle_agreement_geom"].mean())
    random_baseline = 1.0 / 5.0

    passed = bool(
        full_vs_geom["mean_delta"] >= 0.005
        and full_vs_best["mean_delta"] >= 0.0
        and oracle_agreement >= 0.45
        and oracle_agreement >= random_baseline + 0.15
    )

    method_summary.to_csv(root_dir / "selector_summary.csv", index=False)
    agreement_df.to_csv(root_dir / "agreement_summary.csv", index=False)
    summary_df.to_csv(root_dir / "selector_loso.csv", index=False)
    summary = {
        "d1_run_dir": str(d1_run_dir),
        "representation": rep_cfg,
        "window_size": window_size,
        "best_single_expert": best_single_expert,
        "pairwise": {
            "linear_vs_geom": full_vs_geom,
            "linear_vs_best_single": full_vs_best,
        },
        "oracle_agreement_full": oracle_agreement,
        "oracle_agreement_geom": geom_agreement,
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    _save_json(summary, root_dir / "summary.json")

    plt.figure(figsize=(8.0, 4.5))
    plot_df = method_summary.set_index("method").loc[
        ["best_single_expert", "geom_only_selector", "linear_proxy_selector", "oracle_expert"]
    ].reset_index()
    plt.bar(plot_df["method"], plot_df["accuracy_mean"])
    plt.ylabel("Accuracy")
    plt.title("D2 selector comparison")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "selector_comparison.pdf", dpi=300)
    plt.close()

    logger.info("Saved D2 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
