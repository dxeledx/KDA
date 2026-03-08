#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from src.alignment.koopman_alignment import (  # noqa: E402
    KoopmanAffineAligner,
    KoopmanFeatureProjector,
    build_supervised_aligner,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features.covariance import compute_covariances  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d1r")

REPRESENTATION_GRID = [
    {"pca_rank": 16, "lifting": "quadratic"},
    {"pca_rank": 32, "lifting": "quadratic"},
    {"pca_rank": 64, "lifting": "quadratic"},
    {"pca_rank": 16, "lifting": "quadratic_cubic"},
    {"pca_rank": 32, "lifting": "quadratic_cubic"},
    {"pca_rank": 64, "lifting": "quadratic_cubic"},
]
BASE_ALIGNER_CONFIGS = [
    {"method": "A0", "k": None, "reg_lambda": None, "normalize_output": False},
    {"method": "A1", "k": 16, "reg_lambda": 1.0e-3, "normalize_output": False},
    {"method": "A2", "k": 16, "reg_lambda": 1.0e-3, "normalize_output": False},
    {"method": "A3", "k": 16, "reg_lambda": 1.0e-3, "normalize_output": False},
    {"method": "legacy-affine", "k": None, "reg_lambda": None, "normalize_output": False},
]
STABILITY_GRID = [
    {"method": method, "k": k, "reg_lambda": reg_lambda, "normalize_output": normalize_output}
    for method in ("A1", "A2", "A3")
    for k in (8, 16, 32)
    for reg_lambda in (1.0e-4, 1.0e-3, 1.0e-2)
    for normalize_output in (False, True)
]
ALIGNER_COMPLEXITY = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "legacy-affine": 4}


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


def _resolve_targets(all_subjects: Sequence[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def class_distance_ratio(features: np.ndarray, y: np.ndarray, eps: float = 1.0e-8) -> float:
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    if features.ndim != 2 or len(classes) <= 1:
        return 0.0

    centroids = []
    within_terms = []
    for label in classes:
        block = features[y == label]
        if len(block) == 0:
            continue
        mean = block.mean(axis=0)
        centroids.append(mean)
        within_terms.append(float(np.mean(np.sum((block - mean) ** 2, axis=1))))

    if len(centroids) <= 1:
        return 0.0

    between_terms = []
    for idx in range(len(centroids)):
        for jdx in range(idx + 1, len(centroids)):
            between_terms.append(float(np.sum((centroids[idx] - centroids[jdx]) ** 2)))
    between = float(np.mean(between_terms)) if between_terms else 0.0
    within = float(np.mean(within_terms)) if within_terms else 0.0
    return float(between / (within + float(eps)))


def _summarize_against_reference(
    method_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    method_name: str,
    elapsed_sec: float,
    extra: Dict | None = None,
) -> Dict:
    delta = method_df["accuracy"].to_numpy(dtype=np.float64) - reference_df["accuracy"].to_numpy(dtype=np.float64)
    row = {
        "method": method_name,
        "accuracy_mean": float(method_df["accuracy"].mean()),
        "accuracy_std": float(method_df["accuracy"].std(ddof=1)) if len(method_df) > 1 else 0.0,
        "delta_vs_reference": float(delta.mean()),
        "wins_vs_reference": int(np.sum(delta > 0.0)),
        "losses_vs_reference": int(np.sum(delta < 0.0)),
        "draws_vs_reference": int(np.sum(np.isclose(delta, 0.0))),
        "elapsed_sec": float(elapsed_sec),
    }
    if extra:
        row.update(extra)
    return row


def _load_fold_cache(
    loader,
    all_subjects: Sequence[int],
    target_subjects: Sequence[int],
    pre,
    cov_eps: float,
) -> List[Dict[str, np.ndarray | int]]:
    folds = []
    for target_subject in target_subjects:
        X_source_blocks, y_source_blocks = [], []
        for subject in all_subjects:
            if subject == target_subject:
                continue
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

        folds.append(
            {
                "target_subject": int(target_subject),
                "cov_source": compute_covariances(X_source, eps=cov_eps),
                "y_source": np.asarray(y_source, dtype=np.int64),
                "cov_target_train": compute_covariances(X_target_train, eps=cov_eps),
                "cov_target_test": compute_covariances(X_target_test, eps=cov_eps),
                "y_target_test": np.asarray(y_target_test, dtype=np.int64),
            }
        )
    return folds


def _evaluate_representation(
    folds: Sequence[Dict[str, np.ndarray | int]],
    rep_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    rows = []
    for fold in folds:
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold["cov_source"])
        psi_source = projector.transform(fold["cov_source"])
        psi_target_test = projector.transform(fold["cov_target_test"])
        lda = LDA(**lda_kwargs).fit(psi_source, fold["y_source"])
        y_pred = lda.predict(psi_target_test)
        rows.append(
            {
                "target_subject": int(fold["target_subject"]),
                **compute_metrics(fold["y_target_test"], y_pred),
                "distance_ratio": class_distance_ratio(psi_target_test, fold["y_target_test"]),
            }
        )
    return pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True), float(time.perf_counter() - start)


def _fit_static_aligner(
    method: str,
    psi_source: np.ndarray,
    y_source: np.ndarray,
    psi_target_train: np.ndarray,
    cfg: Dict[str, object],
    cov_eps: float,
):
    method = str(method)
    if method == "A0":
        return None
    if method == "legacy-affine":
        return KoopmanAffineAligner(eps=cov_eps).fit(psi_source, psi_target_train)
    return build_supervised_aligner(
        method,
        k=int(cfg["k"]),
        reg_lambda=float(cfg["reg_lambda"]),
        normalize_output=bool(cfg["normalize_output"]),
    ).fit(psi_source, y_source)


def _evaluate_static_config(
    folds: Sequence[Dict[str, np.ndarray | int]],
    rep_cfg: Dict[str, object],
    aligner_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
    cov_eps: float,
    save_details_dir: Path | None = None,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    rows = []
    for fold in folds:
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold["cov_source"])
        psi_source = projector.transform(fold["cov_source"])
        psi_target_train = projector.transform(fold["cov_target_train"])
        psi_target_test = projector.transform(fold["cov_target_test"])
        aligner = _fit_static_aligner(
            str(aligner_cfg["method"]),
            psi_source,
            fold["y_source"],
            psi_target_train,
            aligner_cfg,
            cov_eps,
        )

        psi_source_aligned = psi_source if aligner is None else aligner.transform(psi_source)
        psi_target_test_aligned = psi_target_test if aligner is None else aligner.transform(psi_target_test)
        lda = LDA(**lda_kwargs).fit(psi_source_aligned, fold["y_source"])
        y_pred = lda.predict(psi_target_test_aligned)
        dist_before = class_distance_ratio(psi_target_test, fold["y_target_test"])
        dist_after = class_distance_ratio(psi_target_test_aligned, fold["y_target_test"])
        rows.append(
            {
                "target_subject": int(fold["target_subject"]),
                **compute_metrics(fold["y_target_test"], y_pred),
                "distance_ratio_before": dist_before,
                "distance_ratio_after": dist_after,
                "distance_ratio_delta": float(dist_after - dist_before),
                "fold_failure": 0,
            }
        )

        if save_details_dir is not None:
            np.savez(
                save_details_dir / f"subject_A{int(fold['target_subject']):02d}.npz",
                y_true=np.asarray(fold["y_target_test"], dtype=np.int64),
                y_pred=np.asarray(y_pred, dtype=np.int64),
                psi_raw=np.asarray(psi_target_test, dtype=np.float64),
                psi_aligned=np.asarray(psi_target_test_aligned, dtype=np.float64),
                trial_index=np.arange(len(fold["y_target_test"]), dtype=np.int64),
            )

    return pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True), float(time.perf_counter() - start)


def _representation_summary_row(
    rep_cfg: Dict[str, object],
    method_df: pd.DataFrame,
    noalign_df: pd.DataFrame,
    elapsed_sec: float,
) -> Dict:
    base = _summarize_against_reference(
        method_df,
        noalign_df,
        method_name=f"koopman-noalign-pca{rep_cfg['pca_rank']}-{rep_cfg['lifting']}",
        elapsed_sec=elapsed_sec,
        extra={
            "pca_rank": int(rep_cfg["pca_rank"]),
            "lifting": str(rep_cfg["lifting"]),
            "distance_ratio_mean": float(method_df["distance_ratio"].mean()),
            "distance_ratio_std": float(method_df["distance_ratio"].std(ddof=1)) if len(method_df) > 1 else 0.0,
        },
    )
    return base


def _static_summary_row(
    method_df: pd.DataFrame,
    a0_df: pd.DataFrame,
    aligner_cfg: Dict[str, object],
    elapsed_sec: float,
) -> Dict:
    base = _summarize_against_reference(
        method_df,
        a0_df,
        method_name=str(aligner_cfg["method"]),
        elapsed_sec=elapsed_sec,
        extra={
            "k": aligner_cfg["k"],
            "reg_lambda": aligner_cfg["reg_lambda"],
            "normalize_output": bool(aligner_cfg["normalize_output"]),
            "distance_ratio_before_mean": float(method_df["distance_ratio_before"].mean()),
            "distance_ratio_after_mean": float(method_df["distance_ratio_after"].mean()),
            "distance_ratio_delta_mean": float(method_df["distance_ratio_delta"].mean()),
            "distance_ratio_improved_subjects": int(np.sum(method_df["distance_ratio_delta"].to_numpy(dtype=np.float64) > 0.0)),
            "fold_failures": int(method_df["fold_failure"].sum()),
        },
    )
    return base


def _best_representation(rep_summary: pd.DataFrame) -> pd.Series:
    ranked = rep_summary.sort_values(
        ["accuracy_mean", "distance_ratio_mean", "pca_rank", "lifting"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _best_static_candidate(aligner_summary: pd.DataFrame, stability_summary: pd.DataFrame) -> pd.Series:
    candidates = pd.concat(
        [
            aligner_summary[aligner_summary["method"] == "A0"],
            stability_summary[stability_summary["method"].isin(["A1", "A2", "A3"])],
        ],
        axis=0,
        ignore_index=True,
    ).copy()
    candidates["complexity_rank"] = candidates["method"].map(ALIGNER_COMPLEXITY).fillna(99).astype(int)
    ranked = candidates.sort_values(
        ["accuracy_mean", "wins_vs_reference", "fold_failures", "complexity_rank", "elapsed_sec"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(df[x_col].astype(str), df[y_col].to_numpy(dtype=np.float64))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.evaluation.protocols import evaluate_loso
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--representation-gap-threshold", type=float, default=0.01)
    args = parser.parse_args()

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
    csp_kwargs = model_cfg["csp"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/ksda/exp_d1r/{args.run_name}")
    fig_dir = ensure_dir(root_dir / "figures")
    details_dir = ensure_dir(root_dir / "details")

    logger.info("Building fold cache for targets=%s", target_subjects)
    folds = _load_fold_cache(loader, all_subjects, target_subjects, pre, cov_eps)

    ra_start = time.perf_counter()
    ra_df = evaluate_loso(loader, all_subjects, pre, csp_kwargs, lda_kwargs, "ra", cov_eps)
    ra_elapsed = float(time.perf_counter() - ra_start)
    noalign_start = time.perf_counter()
    noalign_df = evaluate_loso(loader, all_subjects, pre, csp_kwargs, lda_kwargs, "noalign", cov_eps)
    noalign_elapsed = float(time.perf_counter() - noalign_start)
    ra_df = ra_df.loc[ra_df["target_subject"].isin(target_subjects)].sort_values("target_subject").reset_index(drop=True)
    noalign_df = noalign_df.loc[noalign_df["target_subject"].isin(target_subjects)].sort_values("target_subject").reset_index(drop=True)
    ra_df.to_csv(root_dir / "ra_loso.csv", index=False)
    noalign_df.to_csv(root_dir / "noalign_loso.csv", index=False)

    representation_rows = []
    representation_fold_results: Dict[str, pd.DataFrame] = {}
    for rep_cfg in REPRESENTATION_GRID:
        rep_df, elapsed = _evaluate_representation(folds, rep_cfg, lda_kwargs)
        key = f"pca{rep_cfg['pca_rank']}-{rep_cfg['lifting']}"
        representation_fold_results[key] = rep_df
        representation_rows.append(_representation_summary_row(rep_cfg, rep_df, noalign_df, elapsed))
        logger.info("D1-R representation=%s acc=%.4f", key, representation_rows[-1]["accuracy_mean"])
    representation_summary = pd.DataFrame(representation_rows).sort_values(
        ["accuracy_mean", "distance_ratio_mean"], ascending=[False, False]
    )
    representation_summary.to_csv(root_dir / "representation_benchmark.csv", index=False)

    best_rep = _best_representation(representation_summary)
    best_rep_cfg = {"pca_rank": int(best_rep["pca_rank"]), "lifting": str(best_rep["lifting"])}
    best_rep_df = representation_fold_results[f"pca{best_rep_cfg['pca_rank']}-{best_rep_cfg['lifting']}"]
    representation_viable = bool(
        float(noalign_df["accuracy"].mean()) - float(best_rep["accuracy_mean"]) <= float(args.representation_gap_threshold)
    )

    aligner_subject_results = {}
    aligner_elapsed = {}
    aligner_rows = []
    for aligner_cfg in BASE_ALIGNER_CONFIGS:
        method_df, elapsed = _evaluate_static_config(folds, best_rep_cfg, aligner_cfg, lda_kwargs, cov_eps)
        aligner_subject_results[str(aligner_cfg["method"])] = method_df
        aligner_elapsed[str(aligner_cfg["method"])] = elapsed
        logger.info("D1-R aligner=%s acc=%.4f", aligner_cfg["method"], float(method_df["accuracy"].mean()))

    a0_df = aligner_subject_results["A0"]
    for aligner_cfg in BASE_ALIGNER_CONFIGS:
        method_df = aligner_subject_results[str(aligner_cfg["method"])]
        aligner_rows.append(_static_summary_row(method_df, a0_df, aligner_cfg, aligner_elapsed[str(aligner_cfg["method"])]))
    aligner_summary = pd.DataFrame(aligner_rows).sort_values(
        ["accuracy_mean", "wins_vs_reference"], ascending=[False, False]
    )
    aligner_summary.to_csv(root_dir / "aligner_benchmark.csv", index=False)

    stability_rows = []
    stability_subject_results = {}
    for aligner_cfg in STABILITY_GRID:
        method_df, elapsed = _evaluate_static_config(folds, best_rep_cfg, aligner_cfg, lda_kwargs, cov_eps)
        key = (
            f"{aligner_cfg['method']}-k{aligner_cfg['k']}-"
            f"lam{aligner_cfg['reg_lambda']}-norm{int(bool(aligner_cfg['normalize_output']))}"
        )
        stability_subject_results[key] = method_df
        row = _static_summary_row(method_df, a0_df, aligner_cfg, elapsed)
        stability_rows.append(row)
        logger.info("D1-R stability=%s acc=%.4f", key, row["accuracy_mean"])
    stability_summary = pd.DataFrame(stability_rows).sort_values(
        ["accuracy_mean", "wins_vs_reference", "distance_ratio_delta_mean"],
        ascending=[False, False, False],
    )
    stability_summary.to_csv(root_dir / "stability_sweep.csv", index=False)

    best_static = _best_static_candidate(aligner_summary, stability_summary)
    best_static_cfg = {
        "method": str(best_static["method"]),
        "k": None if pd.isna(best_static.get("k")) else int(best_static["k"]),
        "reg_lambda": None if pd.isna(best_static.get("reg_lambda")) else float(best_static["reg_lambda"]),
        "normalize_output": bool(best_static.get("normalize_output", False)),
    }

    best_static_ready = bool(
        float(best_static["accuracy_mean"]) >= 0.42
        and int(best_static["fold_failures"]) == 0
        and float(best_static["distance_ratio_delta_mean"]) >= 0.0
        and int(best_static["distance_ratio_improved_subjects"]) >= max(5, int(math.ceil(len(target_subjects) / 2.0)))
    )

    best_df, _ = _evaluate_static_config(
        folds,
        best_rep_cfg,
        best_static_cfg,
        lda_kwargs,
        cov_eps,
        save_details_dir=details_dir,
    )

    summary = {
        "ra": {
            "accuracy_mean": float(ra_df["accuracy"].mean()),
            "accuracy_std": float(ra_df["accuracy"].std(ddof=1)),
            "elapsed_sec": ra_elapsed,
        },
        "noalign": {
            "accuracy_mean": float(noalign_df["accuracy"].mean()),
            "accuracy_std": float(noalign_df["accuracy"].std(ddof=1)),
            "elapsed_sec": noalign_elapsed,
        },
        "representation": {
            "best": {
                "pca_rank": best_rep_cfg["pca_rank"],
                "lifting": best_rep_cfg["lifting"],
                "accuracy_mean": float(best_rep["accuracy_mean"]),
                "distance_ratio_mean": float(best_rep["distance_ratio_mean"]),
            },
            "representation_viable": representation_viable,
            "gap_vs_noalign": float(best_rep["accuracy_mean"] - float(noalign_df["accuracy"].mean())),
        },
        "best_static": {
            **best_static_cfg,
            "accuracy_mean": float(best_static["accuracy_mean"]),
            "accuracy_std": float(best_static["accuracy_std"]),
            "wins_vs_a0": int(best_static["wins_vs_reference"]),
            "distance_ratio_delta_mean": float(best_static["distance_ratio_delta_mean"]),
            "distance_ratio_improved_subjects": int(best_static["distance_ratio_improved_subjects"]),
            "fold_failures": int(best_static["fold_failures"]),
            "ready_for_d1pr": best_static_ready,
        },
        "gates": {
            "representation_viable": representation_viable,
            "static_ready_for_d1pr": best_static_ready,
        },
        "frozen_reference": {
            "legacy_affine": aligner_summary.loc[aligner_summary["method"] == "legacy-affine"].iloc[0].to_dict(),
        },
    }
    _save_json(summary, root_dir / "summary.json")
    _save_json(
        {
            "representation": best_rep_cfg,
            "best_static": best_static_cfg,
            "gates": summary["gates"],
            "best_static_metrics": summary["best_static"],
        },
        root_dir / "best_static.json",
    )

    _plot_bar(
        representation_summary.assign(label=lambda df: df["pca_rank"].astype(str) + "-" + df["lifting"]),
        "label",
        "accuracy_mean",
        "D1-R representation benchmark",
        fig_dir / "representation_accuracy.pdf",
    )
    _plot_bar(
        aligner_summary.assign(label=lambda df: df["method"]),
        "label",
        "accuracy_mean",
        "D1-R aligner benchmark",
        fig_dir / "aligner_accuracy.pdf",
    )
    logger.info("Saved D1-R outputs to %s", root_dir)


if __name__ == "__main__":
    main()
