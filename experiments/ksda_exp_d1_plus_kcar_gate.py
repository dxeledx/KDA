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

from src.alignment.conditional import LinearConditionalWeight
from src.alignment.koopman_alignment import fit_alignment, transform
from src.evaluation.kcar_analysis import (
    compute_kcar,
    compute_transition_residuals,
    fit_koopman_operator,
    fit_subjectwise_global_koopman,
)
from src.evaluation.metrics import compute_metrics
from src.features.covariance import compute_covariances
from src.models.classifiers import LDA
from src.utils.logger import get_logger


logger = get_logger("ksda_exp_d1_plus")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def kcar_gate(rho_t: float, a: float = 1.0, b: float = 2.0, c: float = 0.0) -> float:
    return float(1.0 / (1.0 + np.exp(-(a - b * float(rho_t) + c))))


def compute_kcar_window(states: np.ndarray, source_operator, target_operator) -> float:
    if len(states) < 2:
        return 0.0
    e_src = compute_transition_residuals(states, source_operator)
    e_tgt = compute_transition_residuals(states, target_operator)
    return compute_kcar(e_src, e_tgt)


def _compute_geometric_context(states: np.ndarray, source_mean: np.ndarray, recent_window: int = 5) -> np.ndarray:
    target_mean = None
    history = []
    rows = []
    for state in states:
        d_src = float(np.linalg.norm(state - source_mean))
        if target_mean is None:
            d_tgt = 0.0
            target_mean = state.copy()
        else:
            d_tgt = float(np.linalg.norm(state - target_mean))
            target_mean = 0.9 * target_mean + 0.1 * state
        sigma_recent = 0.0
        if len(history) >= recent_window:
            recent = np.stack(history[-recent_window:], axis=0)
            sigma_recent = float(np.std(recent))
        rows.append(np.asarray([d_src, d_tgt, sigma_recent], dtype=np.float64))
        history.append(state.copy())
    return np.asarray(rows, dtype=np.float64)


def _fit_context_stats(source_state_blocks: List[np.ndarray], recent_window: int = 5) -> Dict[str, np.ndarray]:
    source_mean = np.mean(np.concatenate(source_state_blocks, axis=0), axis=0)
    contexts = []
    for states in source_state_blocks:
        contexts.append(_compute_geometric_context(states, source_mean, recent_window=recent_window))
    stacked = np.concatenate(contexts, axis=0)
    return {"mean": stacked.mean(axis=0), "std": np.clip(stacked.std(axis=0), 1.0e-8, None), "source_mean": source_mean}


def _normalize_context(context: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (context - stats["mean"]) / stats["std"]


def _run_fold(
    loader,
    all_subjects: List[int],
    target_subject: int,
    pre,
    lda_kwargs: Dict,
    cov_eps: float,
    pca_rank: int,
) -> Dict[str, object]:
    X_source_blocks, y_source_blocks = [], []
    for subject in all_subjects:
        if subject == target_subject:
            continue
        X_train_subject, y_train_subject = loader.load_subject(subject, split="train")
        X_source_blocks.append(X_train_subject)
        y_source_blocks.append(y_train_subject)

    X_source_raw = np.concatenate(X_source_blocks, axis=0)
    y_source = np.concatenate(y_source_blocks, axis=0)
    X_target_train_raw, _ = loader.load_subject(target_subject, split="train")
    X_target_test_raw, y_target_test = loader.load_subject(target_subject, split="test")

    X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
    X_target_train = pre.transform(X_target_train_raw)
    X_target_test = pre.transform(X_target_test_raw)

    cov_source = compute_covariances(X_source, eps=cov_eps)
    cov_target_train = compute_covariances(X_target_train, eps=cov_eps)
    cov_target_test = compute_covariances(X_target_test, eps=cov_eps)

    projector, aligner, psi_source, _ = fit_alignment(
        cov_source, cov_target_train, pca_rank=pca_rank, cov_eps=cov_eps
    )
    psi_target_test = transform(cov_target_test, projector, None)
    psi_target_test_aligned = aligner.transform(psi_target_test)
    z_source = projector.transform_tangent(cov_source)
    z_target_train = projector.transform_tangent(cov_target_train)
    z_target_test = projector.transform_tangent(cov_target_test)

    source_state_blocks = []
    start = 0
    for block in X_source_blocks:
        n = block.shape[0]
        source_state_blocks.append(z_source[start : start + n])
        start += n

    source_operator = fit_subjectwise_global_koopman(source_state_blocks, ridge_alpha=1.0e-3)
    target_operator = fit_koopman_operator(z_target_train, ridge_alpha=1.0e-3)
    stats = _fit_context_stats(source_state_blocks)
    geom_raw = _compute_geometric_context(z_target_test, source_mean=stats["source_mean"])
    geom_norm = _normalize_context(geom_raw, stats)
    geom_gate = LinearConditionalWeight(weights=[1.2, 0.6, 0.3], bias=0.0, temperature=1.0, ema_smooth_alpha=0.0)

    lda = LDA(**lda_kwargs).fit(psi_source, y_source)
    y_pred_static = lda.predict(psi_target_test_aligned).astype(np.int64)
    y_pred_fixed05 = lda.predict(0.5 * psi_target_test + 0.5 * psi_target_test_aligned).astype(np.int64)

    warmup = 8
    rho_t, w_t_kcar, w_t_geometric, y_pred_kcar, y_pred_geometric = [], [], [], [], []
    for trial_idx in range(len(y_target_test)):
        if trial_idx < warmup:
            rho = 0.0
            wk = 1.0
            wg = 1.0
        else:
            history_states = z_target_test[max(0, trial_idx - 32) : trial_idx]
            rho = compute_kcar_window(history_states, source_operator, target_operator)
            wk = kcar_gate(rho)
            wg = float(geom_gate.predict(geom_norm[trial_idx]))
        psi_k = (1.0 - wk) * psi_target_test[trial_idx] + wk * psi_target_test_aligned[trial_idx]
        psi_g = (1.0 - wg) * psi_target_test[trial_idx] + wg * psi_target_test_aligned[trial_idx]
        y_pred_kcar.append(int(lda.predict(psi_k[None, ...])[0]))
        y_pred_geometric.append(int(lda.predict(psi_g[None, ...])[0]))
        rho_t.append(float(rho))
        w_t_kcar.append(float(wk))
        w_t_geometric.append(float(wg))

    return {
        "target_subject": target_subject,
        "y_true": np.asarray(y_target_test, dtype=np.int64),
        "y_pred_static": y_pred_static,
        "y_pred_fixed05": y_pred_fixed05,
        "y_pred_kcar": np.asarray(y_pred_kcar, dtype=np.int64),
        "y_pred_geometric": np.asarray(y_pred_geometric, dtype=np.int64),
        "rho_t": np.asarray(rho_t, dtype=np.float64),
        "w_t_kcar": np.asarray(w_t_kcar, dtype=np.float64),
        "w_t_geometric": np.asarray(w_t_geometric, dtype=np.float64),
        "d_src": geom_raw[:, 0],
        "d_tgt": geom_raw[:, 1],
        "sigma_recent": geom_raw[:, 2],
        "trial_index": np.arange(len(y_target_test), dtype=np.int64),
    }


def _summarize(df: pd.DataFrame, ra_df: pd.DataFrame, method_name: str, elapsed_sec: float) -> Dict:
    delta = df["accuracy"].to_numpy(dtype=np.float64) - ra_df["accuracy"].to_numpy(dtype=np.float64)
    return {
        "method": method_name,
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=1)) if len(df) > 1 else 0.0,
        "delta_vs_ra": float(delta.mean()),
        "wins": int(np.sum(delta > 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "draws": int(np.sum(np.isclose(delta, 0.0))),
        "elapsed_sec": elapsed_sec,
    }


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.evaluation.protocols import evaluate_loso
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--pca-rank", type=int, default=16)
    args = parser.parse_args()

    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))
    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    all_subjects = list(map(int, data_cfg["dataset"]["subjects"]))
    target_subjects = [subject for subject in all_subjects if subject in {int(t.strip()) for t in args.targets.split(',') if t.strip()}] if args.targets else list(all_subjects)
    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))
    lda_kwargs = model_cfg["lda"]
    csp_kwargs = model_cfg["csp"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/ksda/exp_d1_plus/{args.run_name}")
    details_dir = ensure_dir(root_dir / "details")
    fig_dir = ensure_dir(root_dir / "figures")

    ra_start = time.perf_counter()
    ra_df = evaluate_loso(loader, all_subjects, pre, csp_kwargs, lda_kwargs, "ra", cov_eps)
    ra_elapsed = float(time.perf_counter() - ra_start)

    methods = {"ksda-static": [], "ksda-geometric-gate": [], "ksda-kcar-gate": [], "fixed_w=0.5": []}
    start = time.perf_counter()
    for target in target_subjects:
        fold = _run_fold(loader, all_subjects, target, pre, lda_kwargs, cov_eps, int(args.pca_rank))
        y_true = fold["y_true"]
        for method_name, pred_key in {
            "ksda-static": "y_pred_static",
            "ksda-geometric-gate": "y_pred_geometric",
            "ksda-kcar-gate": "y_pred_kcar",
            "fixed_w=0.5": "y_pred_fixed05",
        }.items():
            methods[method_name].append({"target_subject": target, **compute_metrics(y_true, fold[pred_key])})
        np.savez(details_dir / f"subject_A{target:02d}.npz", **fold)
        logger.info(
            "D1+ target=%s static=%.4f geom=%.4f kcar=%.4f",
            target,
            methods["ksda-static"][-1]["accuracy"],
            methods["ksda-geometric-gate"][-1]["accuracy"],
            methods["ksda-kcar-gate"][-1]["accuracy"],
        )
    elapsed = float(time.perf_counter() - start)

    comparison_rows = [_summarize(ra_df, ra_df, "ra", ra_elapsed)]
    loso_rows = []
    for method_name in ["ksda-static", "ksda-geometric-gate", "ksda-kcar-gate", "fixed_w=0.5"]:
        df = pd.DataFrame(methods[method_name]).sort_values("target_subject").reset_index(drop=True)
        comparison_rows.append(_summarize(df, ra_df, method_name, elapsed))
        for row in methods[method_name]:
            loso_rows.append({"method": method_name, **row})

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["accuracy_mean", "wins"], ascending=[False, False])
    comparison_df.to_csv(root_dir / "comparison.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(root_dir / "loso_results.csv", index=False)
    _save_json({row["method"]: row for row in comparison_rows}, root_dir / "summary.json")
    _save_json({"ra": ra_elapsed, "phase_total": elapsed}, root_dir / "timing.json")

    plt.figure(figsize=(8.5, 4.5))
    order = ["ra", "ksda-static", "ksda-geometric-gate", "ksda-kcar-gate", "fixed_w=0.5"]
    data = [ra_df["accuracy"].to_numpy(dtype=np.float64)]
    loso_df = pd.DataFrame(loso_rows)
    for method in order[1:]:
        data.append(loso_df.loc[loso_df["method"] == method, "accuracy"].to_numpy(dtype=np.float64))
    plt.boxplot(data, labels=order)
    plt.ylabel("Accuracy")
    plt.title("KSDA D.1+ vs baselines")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_comparison.pdf", dpi=300)
    plt.close()
    logger.info("Saved D.1+ outputs to %s", root_dir)


if __name__ == "__main__":
    main()
