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

from src.alignment.conditional import LinearConditionalWeight  # noqa: E402
from src.alignment.koopman_alignment import (  # noqa: E402
    KoopmanFeatureProjector,
    build_supervised_aligner,
)
from src.evaluation.kcar_analysis import (  # noqa: E402
    compute_kcar,
    compute_transition_residuals,
    fit_koopman_operator,
    fit_subjectwise_global_koopman,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.features.covariance import compute_covariances  # noqa: E402
from src.models.classifiers import LDA  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


logger = get_logger("ksda_exp_d1plus_r")


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


def _load_fold_cache(
    loader,
    all_subjects: Sequence[int],
    target_subjects: Sequence[int],
    pre,
    cov_eps: float,
) -> List[Dict[str, np.ndarray | int | list[np.ndarray]]]:
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
                "source_block_lengths": [block.shape[0] for block in X_source_blocks],
            }
        )
    return folds


def kcar_gate(rho_t: float, a: float = 1.0, b: float = 2.0, c: float = 0.0) -> float:
    return float(1.0 / (1.0 + np.exp(-(a - b * float(rho_t) + c))))


def compute_kcar_window(states: np.ndarray, source_operator, target_operator) -> float:
    if len(states) < 2:
        return 0.0
    e_src = compute_transition_residuals(states, source_operator)
    e_tgt = compute_transition_residuals(states, target_operator)
    return compute_kcar(e_src, e_tgt)


def compute_window_kcar_weights(
    rho_t: np.ndarray,
    window_size: int = 32,
    low_weight: float = 0.5,
    high_weight: float = 1.0,
) -> np.ndarray:
    rho_t = np.asarray(rho_t, dtype=np.float64)
    weights = np.full(rho_t.shape[0], float(high_weight), dtype=np.float64)
    for start in range(window_size, rho_t.shape[0], window_size):
        prev_start = start - window_size
        prev_end = start
        current_end = min(start + window_size, rho_t.shape[0])
        rho_window = float(np.mean(rho_t[prev_start:prev_end]))
        weights[start:current_end] = float(low_weight if rho_window > 0.0 else high_weight)
    return weights


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
    contexts = [
        _compute_geometric_context(states, source_mean, recent_window=recent_window)
        for states in source_state_blocks
    ]
    stacked = np.concatenate(contexts, axis=0)
    return {
        "mean": stacked.mean(axis=0),
        "std": np.clip(stacked.std(axis=0), 1.0e-8, None),
        "source_mean": source_mean,
    }


def _normalize_context(context: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (context - stats["mean"]) / stats["std"]


def _fit_best_static_aligner(psi_source: np.ndarray, psi_target_train: np.ndarray, y_source: np.ndarray, best_static: Dict[str, object]):
    method = str(best_static["method"])
    if method == "A0":
        return None
    return build_supervised_aligner(
        method,
        k=int(best_static["k"]),
        reg_lambda=float(best_static["reg_lambda"]),
        normalize_output=bool(best_static["normalize_output"]),
    ).fit(psi_source, y_source)


def _run_fold(
    fold: Dict[str, np.ndarray | int | list[int]],
    best_rep: Dict[str, object],
    best_static: Dict[str, object],
    lda_kwargs: Dict[str, object],
) -> Dict[str, np.ndarray | int]:
    projector = KoopmanFeatureProjector(
        pca_rank=int(best_rep["pca_rank"]),
        lifting=str(best_rep["lifting"]),
    ).fit(fold["cov_source"])

    psi_source = projector.transform(fold["cov_source"])
    psi_target_train = projector.transform(fold["cov_target_train"])
    psi_target_test = projector.transform(fold["cov_target_test"])
    z_source = projector.transform_tangent(fold["cov_source"])
    z_target_train = projector.transform_tangent(fold["cov_target_train"])
    z_target_test = projector.transform_tangent(fold["cov_target_test"])

    aligner = _fit_best_static_aligner(psi_source, psi_target_train, fold["y_source"], best_static)
    psi_source_aligned = psi_source if aligner is None else aligner.transform(psi_source)
    psi_target_test_aligned = psi_target_test if aligner is None else aligner.transform(psi_target_test)

    lda = LDA(**lda_kwargs).fit(psi_source_aligned, fold["y_source"])
    y_pred_static = lda.predict(psi_target_test_aligned).astype(np.int64)

    source_blocks = []
    start = 0
    for length in fold["source_block_lengths"]:
        source_blocks.append(z_source[start : start + int(length)])
        start += int(length)

    source_operator = fit_subjectwise_global_koopman(source_blocks, ridge_alpha=1.0e-3)
    target_operator = fit_koopman_operator(z_target_train, ridge_alpha=1.0e-3)
    context_stats = _fit_context_stats(source_blocks)
    geom_raw = _compute_geometric_context(z_target_test, context_stats["source_mean"])
    geom_norm = _normalize_context(geom_raw, context_stats)
    geom_gate = LinearConditionalWeight(
        weights=[1.2, 0.6, 0.3],
        bias=0.0,
        temperature=1.0,
        ema_smooth_alpha=0.0,
    )

    warmup = 8
    rho_t = np.zeros(len(fold["y_target_test"]), dtype=np.float64)
    w_t_kcar = np.ones(len(fold["y_target_test"]), dtype=np.float64)
    w_t_geom = np.ones(len(fold["y_target_test"]), dtype=np.float64)
    y_pred_kcar, y_pred_geom = [], []
    for trial_idx in range(len(fold["y_target_test"])):
        if trial_idx >= warmup:
            history_states = z_target_test[max(0, trial_idx - 32) : trial_idx]
            rho_t[trial_idx] = compute_kcar_window(history_states, source_operator, target_operator)
            w_t_kcar[trial_idx] = kcar_gate(rho_t[trial_idx])
            w_t_geom[trial_idx] = float(geom_gate.predict(geom_norm[trial_idx]))

        psi_k = (1.0 - w_t_kcar[trial_idx]) * psi_target_test[trial_idx] + w_t_kcar[trial_idx] * psi_target_test_aligned[trial_idx]
        psi_g = (1.0 - w_t_geom[trial_idx]) * psi_target_test[trial_idx] + w_t_geom[trial_idx] * psi_target_test_aligned[trial_idx]
        y_pred_kcar.append(int(lda.predict(psi_k[None, ...])[0]))
        y_pred_geom.append(int(lda.predict(psi_g[None, ...])[0]))

    w_t_window = compute_window_kcar_weights(rho_t, window_size=32, low_weight=0.5, high_weight=1.0)
    y_pred_window = []
    for trial_idx in range(len(fold["y_target_test"])):
        psi_w = (1.0 - w_t_window[trial_idx]) * psi_target_test[trial_idx] + w_t_window[trial_idx] * psi_target_test_aligned[trial_idx]
        y_pred_window.append(int(lda.predict(psi_w[None, ...])[0]))

    return {
        "target_subject": int(fold["target_subject"]),
        "y_true": np.asarray(fold["y_target_test"], dtype=np.int64),
        "y_pred_static": y_pred_static,
        "y_pred_geometric": np.asarray(y_pred_geom, dtype=np.int64),
        "y_pred_kcar": np.asarray(y_pred_kcar, dtype=np.int64),
        "y_pred_window": np.asarray(y_pred_window, dtype=np.int64),
        "rho_t": rho_t,
        "w_t_kcar": w_t_kcar,
        "w_t_geometric": w_t_geom,
        "w_t_window": w_t_window,
        "d_src": geom_raw[:, 0],
        "d_tgt": geom_raw[:, 1],
        "sigma_recent": geom_raw[:, 2],
        "trial_index": np.arange(len(fold["y_target_test"]), dtype=np.int64),
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


def _pairwise_summary(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> Dict:
    delta = left["accuracy"].to_numpy(dtype=np.float64) - right["accuracy"].to_numpy(dtype=np.float64)
    return {
        "left": left_name,
        "right": right_name,
        "mean_delta": float(delta.mean()),
        "wins": int(np.sum(delta > 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "draws": int(np.sum(np.isclose(delta, 0.0))),
        "worst_delta": float(delta.min()),
        "best_delta": float(delta.max()),
    }


def main() -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor
    from src.evaluation.protocols import evaluate_loso
    from src.utils.config import ensure_dir, load_yaml, seed_everything

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--d1r-run-dir", required=True)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    d1r_dir = Path(args.d1r_run_dir)
    best_static_info = json.loads((d1r_dir / "best_static.json").read_text(encoding="utf-8"))
    if (not bool(best_static_info["gates"]["static_ready_for_d1pr"])) and (not args.force):
        raise RuntimeError("D1-R best-static did not pass the D1+-R entry gate. Use --force to override.")

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

    root_dir = ensure_dir(f"results/ksda/exp_d1plus_r/{args.run_name}")
    details_dir = ensure_dir(root_dir / "details")
    fig_dir = ensure_dir(root_dir / "figures")

    folds = _load_fold_cache(loader, all_subjects, target_subjects, pre, cov_eps)

    ra_start = time.perf_counter()
    ra_df = evaluate_loso(loader, all_subjects, pre, csp_kwargs, lda_kwargs, "ra", cov_eps)
    ra_elapsed = float(time.perf_counter() - ra_start)
    ra_df = ra_df.loc[ra_df["target_subject"].isin(target_subjects)].sort_values("target_subject").reset_index(drop=True)

    methods = {"best-static": [], "geom-gate": [], "kcar-gate": [], "window-kcar": []}
    start = time.perf_counter()
    for fold in folds:
        fold_result = _run_fold(
            fold,
            best_static_info["representation"],
            best_static_info["best_static"],
            lda_kwargs,
        )
        np.savez(details_dir / f"subject_A{int(fold['target_subject']):02d}.npz", **fold_result)
        y_true = fold_result["y_true"]
        methods["best-static"].append({"target_subject": int(fold["target_subject"]), **compute_metrics(y_true, fold_result["y_pred_static"])})
        methods["geom-gate"].append({"target_subject": int(fold["target_subject"]), **compute_metrics(y_true, fold_result["y_pred_geometric"])})
        methods["kcar-gate"].append({"target_subject": int(fold["target_subject"]), **compute_metrics(y_true, fold_result["y_pred_kcar"])})
        methods["window-kcar"].append({"target_subject": int(fold["target_subject"]), **compute_metrics(y_true, fold_result["y_pred_window"])})
        logger.info(
            "D1+-R target=%s best=%.4f geom=%.4f kcar=%.4f window=%.4f",
            int(fold["target_subject"]),
            methods["best-static"][-1]["accuracy"],
            methods["geom-gate"][-1]["accuracy"],
            methods["kcar-gate"][-1]["accuracy"],
            methods["window-kcar"][-1]["accuracy"],
        )
    elapsed = float(time.perf_counter() - start)

    comparison_rows = [_summarize(ra_df, ra_df, "ra", ra_elapsed)]
    loso_rows = []
    method_dfs = {}
    for method_name in ["best-static", "geom-gate", "kcar-gate", "window-kcar"]:
        df = pd.DataFrame(methods[method_name]).sort_values("target_subject").reset_index(drop=True)
        method_dfs[method_name] = df
        comparison_rows.append(_summarize(df, ra_df, method_name, elapsed))
        for row in methods[method_name]:
            loso_rows.append({"method": method_name, **row})

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["accuracy_mean", "wins"], ascending=[False, False])
    comparison_df.to_csv(root_dir / "comparison.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(root_dir / "loso_results.csv", index=False)

    metric_table = {row["method"]: row for row in comparison_rows}
    kcar_vs_geom = _pairwise_summary(method_dfs["kcar-gate"], method_dfs["geom-gate"], "kcar-gate", "geom-gate")
    kcar_vs_best = _pairwise_summary(method_dfs["kcar-gate"], method_dfs["best-static"], "kcar-gate", "best-static")
    window_vs_kcar = _pairwise_summary(method_dfs["window-kcar"], method_dfs["kcar-gate"], "window-kcar", "kcar-gate")
    signal_success = bool(
        kcar_vs_geom["mean_delta"] >= 0.005
        and kcar_vs_geom["wins"] >= kcar_vs_geom["losses"]
    )
    method_success = bool(kcar_vs_best["mean_delta"] >= 0.0)
    summary = {
        **metric_table,
        "pairwise": {
            "kcar_vs_geom": kcar_vs_geom,
            "kcar_vs_best_static": kcar_vs_best,
            "window_vs_kcar": window_vs_kcar,
        },
        "gates": {
            "signal_success": signal_success,
            "method_success": method_success,
            "ready_for_d2_d3": bool(
                best_static_info["best_static_metrics"]["accuracy_mean"] >= 0.42
                and best_static_info["best_static_metrics"]["accuracy_mean"] >= metric_table["ra"]["accuracy_mean"] - 0.01
                and signal_success
                and method_success
            ),
        },
        "best_static_source": best_static_info,
    }
    _save_json(summary, root_dir / "summary.json")
    _save_json({"ra": ra_elapsed, "phase_total": elapsed}, root_dir / "timing.json")

    plt.figure(figsize=(9.0, 4.8))
    order = ["ra", "best-static", "geom-gate", "kcar-gate", "window-kcar"]
    loso_df = pd.DataFrame(loso_rows)
    data = [ra_df["accuracy"].to_numpy(dtype=np.float64)]
    for method in order[1:]:
        data.append(loso_df.loc[loso_df["method"] == method, "accuracy"].to_numpy(dtype=np.float64))
    plt.boxplot(data, labels=order)
    plt.ylabel("Accuracy")
    plt.title("KSDA D1+-R signal benchmark")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_comparison.pdf", dpi=300)
    plt.close()
    logger.info("Saved D1+-R outputs to %s", root_dir)


if __name__ == "__main__":
    main()
