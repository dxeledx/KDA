#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.utils.mean import mean_riemann

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.behavior_feedback import BehaviorGuidedFeedback
from src.alignment.conditional import FixedWeight, LinearConditionalWeight
from src.alignment.dca_bgf import DCABGF
from src.alignment.euclidean import apply_alignment, compute_alignment_matrix
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.evaluation.stage2_analysis import pair_subject_deltas, sliding_window_mean, summarize_against_reference
from src.features.covariance import compute_covariances, mean_covariance
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.context import ContextComputer
from src.utils.logger import get_logger


logger = get_logger("stage2_rescue")


def _deep_merge(base: Dict, overrides: Optional[Dict]) -> Dict:
    merged = copy.deepcopy(base)
    if overrides is None:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label)
    return safe.strip("_") or "candidate"


def _resolve_targets(all_subjects: Sequence[int], targets: Optional[str]) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = [int(tok.strip()) for tok in targets.split(",") if tok.strip()]
    return [subject for subject in all_subjects if subject in wanted]


def _resolve_focus_subjects(
    explicit: Optional[str],
    base_negative_subjects: Iterable[int],
    all_subjects: Sequence[int],
) -> List[int]:
    if explicit:
        return _resolve_targets(all_subjects, explicit)
    focus = {2}
    focus.update(int(subject) for subject in base_negative_subjects)
    return [subject for subject in all_subjects if subject in focus]


def _compute_ra_matrix(X_source: np.ndarray, X_target: np.ndarray, eps: float) -> np.ndarray:
    C_source = mean_riemann(compute_covariances(X_source, eps=eps))
    C_target = mean_riemann(compute_covariances(X_target, eps=eps))
    return compute_alignment_matrix(C_source, C_target, eps=eps)


def _resolve_conditional_weights(
    dca_cfg: Dict,
    context_dim: int,
) -> List[float]:
    cond_cfg = dca_cfg.get("conditional", {})
    weights = list(cond_cfg.get("weights", [1.2, 0.6, 0.3]))
    if len(weights) < context_dim:
        weights = weights + [weights[0]] * (context_dim - len(weights))
    elif len(weights) > context_dim:
        weights = weights[:context_dim]
    return [float(weight) for weight in weights]


def _build_dca_bgf(
    csp: CSP,
    lda: LDA,
    alignment_matrix: np.ndarray,
    source_features: np.ndarray,
    source_covariances: np.ndarray,
    dca_cfg: Dict,
) -> DCABGF:
    ctx_dim = int(dca_cfg.get("context_dim", 3))
    ctx_cfg = dca_cfg.get("context", {})
    normalize = bool(ctx_cfg.get("normalize", True))
    feature_names = ctx_cfg.get("features")
    source_mean = np.mean(source_features, axis=0)
    context_computer = ContextComputer(
        source_mean=source_mean,
        context_dim=ctx_dim,
        feature_names=feature_names,
        ema_alpha=float(ctx_cfg.get("ema_alpha", 0.1)),
        recent_window=int(ctx_cfg.get("recent_window", 5)),
        normalize=normalize,
        source_covariance=mean_riemann(source_covariances),
        cov_eps=float(ctx_cfg.get("cov_eps", 1.0e-6)),
    )
    if normalize:
        covariance_stream = source_covariances if context_computer.requires_trial_covariance else None
        context_computer.fit_normalizer(source_features, covariance_stream=covariance_stream)

    cond_cfg = dca_cfg.get("conditional", {})
    cond_type = str(cond_cfg.get("type", "linear")).lower()
    if cond_type == "linear":
        conditional = LinearConditionalWeight(
            weights=_resolve_conditional_weights(dca_cfg, context_computer.context_dim),
            bias=float(cond_cfg.get("bias", 0.0)),
            temperature=float(cond_cfg.get("temperature", 1.0)),
            ema_smooth_alpha=float(cond_cfg.get("ema_smooth_alpha", 0.2)),
        )
    elif cond_type == "fixed":
        conditional = FixedWeight(float(cond_cfg.get("weight", 0.5)))
    else:
        raise ValueError(f"Unsupported conditional.type: {cond_type}")

    fb_cfg = dca_cfg.get("feedback", {})
    feedback = BehaviorGuidedFeedback(
        window_size=int(fb_cfg.get("window_size", 10)),
        entropy_high_factor=float(fb_cfg.get("entropy_high_factor", 0.8)),
        entropy_low_factor=float(fb_cfg.get("entropy_low_factor", 0.3)),
        conf_trend_threshold=float(fb_cfg.get("conf_trend_threshold", -0.05)),
        alpha=float(fb_cfg.get("alpha", 0.1)),
        beta=float(fb_cfg.get("beta", 0.05)),
        conf_trend_alpha=float(fb_cfg.get("conf_trend_alpha", 0.15)),
        momentum=float(fb_cfg.get("momentum", 0.7)),
        delta_w_max=float(fb_cfg.get("delta_w_max", 0.2)),
        update_every=int(fb_cfg.get("update_every", 1)),
        conflict_mode=str(fb_cfg.get("conflict_mode", "sum")),
    )
    use_feedback = bool(fb_cfg.get("enabled", True))

    return DCABGF(
        csp=csp,
        lda=lda,
        alignment_matrix=alignment_matrix,
        context_computer=context_computer,
        conditional_weight=conditional,
        behavior_feedback=feedback,
        use_feedback=use_feedback,
    )


def _run_adaptive_loso(
    loader: BCIDataLoader,
    all_subjects: Sequence[int],
    target_subjects: Sequence[int],
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    cov_eps: float,
    base_model_cfg: Dict,
    method_name: str,
    dca_overrides: Optional[Dict],
    detail_subjects: Optional[Sequence[int]],
    details_dir: Optional[Path],
) -> Tuple[pd.DataFrame, float, Dict[int, Path]]:
    dca_cfg = _deep_merge(base_model_cfg.get("dca_bgf", {}), dca_overrides or {})
    rows: List[Dict] = []
    saved_details: Dict[int, Path] = {}
    start = time.perf_counter()

    for target in target_subjects:
        X_train_list, y_train_list = [], []
        for sid in all_subjects:
            if sid == target:
                continue
            X_s, y_s = loader.load_subject(sid, split="train")
            X_train_list.append(X_s)
            y_train_list.append(y_s)

        X_source = np.concatenate(X_train_list, axis=0)
        y_source = np.concatenate(y_train_list, axis=0)

        X_t_train, _ = loader.load_subject(target, split="train")
        X_test, y_test = loader.load_subject(target, split="test")

        X_source = pre.fit(X_source, y_source).transform(X_source)
        X_t_train = pre.transform(X_t_train)
        X_test = pre.transform(X_test)

        alignment_matrix = _compute_ra_matrix(X_source, X_t_train, eps=cov_eps)
        source_covariances = compute_covariances(X_source, eps=cov_eps)

        csp = CSP(**csp_kwargs)
        feats_source = csp.fit_transform(X_source, y_source)
        lda = LDA(**lda_kwargs).fit(feats_source, y_source)

        dca = _build_dca_bgf(
            csp=csp,
            lda=lda,
            alignment_matrix=alignment_matrix,
            source_features=feats_source,
            source_covariances=source_covariances,
            dca_cfg=dca_cfg,
        )

        collect_features = detail_subjects is not None and target in detail_subjects
        inference_start = time.perf_counter()
        y_pred, details = dca.predict_online(
            X_test,
            y_true=y_test,
            return_details=True,
            return_features=collect_features,
        )
        inference_sec = float(time.perf_counter() - inference_start)

        if collect_features and details_dir is not None:
            detail_path = details_dir / f"{_sanitize_label(method_name)}_targetA{target:02d}.npz"
            np.savez(detail_path, y_true=y_test, y_pred=y_pred, **details)
            saved_details[target] = detail_path

        metrics = compute_metrics(y_test, y_pred)
        w = np.asarray(details["w"], dtype=np.float64)
        rows.append(
            {
                "target_subject": target,
                "method": method_name,
                **metrics,
                "w_mean": float(w.mean()) if w.size else float("nan"),
                "w_std": float(w.std()) if w.size else float("nan"),
                "inference_sec": inference_sec,
            }
        )
        logger.info(
            "method=%s target=%s acc=%.4f w_mean=%.4f",
            method_name,
            target,
            metrics["accuracy"],
            float(w.mean()) if w.size else float("nan"),
        )

    elapsed_sec = float(time.perf_counter() - start)
    df = pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)
    return df, elapsed_sec, saved_details


def _run_static_loso(
    loader: BCIDataLoader,
    all_subjects: Sequence[int],
    target_subjects: Sequence[int],
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    method: str,
    cov_eps: float,
) -> Tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    rows: List[Dict] = []
    for target in target_subjects:
        X_train_list, y_train_list = [], []
        for sid in all_subjects:
            if sid == target:
                continue
            X_s, y_s = loader.load_subject(sid, split="train")
            X_train_list.append(X_s)
            y_train_list.append(y_s)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_t_train, _ = loader.load_subject(target, split="train")
        X_test, y_test = loader.load_subject(target, split="test")

        X_train = pre.fit(X_train, y_train).transform(X_train)
        X_t_train = pre.transform(X_t_train)
        X_test = pre.transform(X_test)

        if method == "ea":
            C_source = mean_covariance(compute_covariances(X_train, eps=cov_eps))
            C_target = mean_covariance(compute_covariances(X_t_train, eps=cov_eps))
            alignment_matrix = compute_alignment_matrix(C_source, C_target, eps=cov_eps)
            X_eval = apply_alignment(X_test, alignment_matrix)
        elif method == "ra":
            C_source = mean_riemann(compute_covariances(X_train, eps=cov_eps))
            C_target = mean_riemann(compute_covariances(X_t_train, eps=cov_eps))
            alignment_matrix = compute_alignment_matrix(C_source, C_target, eps=cov_eps)
            X_eval = apply_alignment(X_test, alignment_matrix)
        else:
            X_eval = X_test

        csp = CSP(**csp_kwargs)
        feats_train = csp.fit_transform(X_train, y_train)
        lda = LDA(**lda_kwargs).fit(feats_train, y_train)
        y_pred = lda.predict(csp.transform(X_eval))
        rows.append({"target_subject": target, **compute_metrics(y_test, y_pred)})

    df = pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)
    elapsed_sec = float(time.perf_counter() - start)
    df["method"] = method
    return df, elapsed_sec


def _run_ra_detail_targets(
    loader: BCIDataLoader,
    all_subjects: Sequence[int],
    targets: Sequence[int],
    pre: Preprocessor,
    csp_kwargs: Dict,
    lda_kwargs: Dict,
    cov_eps: float,
    details_dir: Path,
) -> Dict[int, Path]:
    saved: Dict[int, Path] = {}
    for target in targets:
        X_train_list, y_train_list = [], []
        for sid in all_subjects:
            if sid == target:
                continue
            X_s, y_s = loader.load_subject(sid, split="train")
            X_train_list.append(X_s)
            y_train_list.append(y_s)

        X_source = np.concatenate(X_train_list, axis=0)
        y_source = np.concatenate(y_train_list, axis=0)
        X_t_train, _ = loader.load_subject(target, split="train")
        X_test, y_test = loader.load_subject(target, split="test")

        X_source = pre.fit(X_source, y_source).transform(X_source)
        X_t_train = pre.transform(X_t_train)
        X_test = pre.transform(X_test)

        alignment_matrix = _compute_ra_matrix(X_source, X_t_train, eps=cov_eps)
        X_eval = apply_alignment(X_test, alignment_matrix)

        csp = CSP(**csp_kwargs)
        feats_source = csp.fit_transform(X_source, y_source)
        lda = LDA(**lda_kwargs).fit(feats_source, y_source)
        y_pred = lda.predict(csp.transform(X_eval))

        detail_path = details_dir / f"ra_targetA{target:02d}.npz"
        np.savez(
            detail_path,
            y_true=y_test,
            y_pred=y_pred,
            correct=(y_pred == y_test).astype(np.int64),
        )
        saved[target] = detail_path
    return saved


def _candidate_record(
    stage: str,
    name: str,
    overrides: Dict,
    df: pd.DataFrame,
    ra_df: pd.DataFrame,
    elapsed_sec: float,
) -> Dict:
    summary = summarize_against_reference(df, ra_df, metric="accuracy")
    return {
        "stage": stage,
        "candidate": name,
        "accuracy_mean": summary["mean"],
        "accuracy_std": summary["std"],
        "delta_vs_ra": summary["delta_vs_reference"],
        "wins": summary["wins"],
        "losses": summary["losses"],
        "draws": summary["draws"],
        "p_value": summary["p_value"],
        "effect_size": summary["effect_size"],
        "delta_ci_low": summary["delta_ci_low"],
        "delta_ci_high": summary["delta_ci_high"],
        "elapsed_sec": elapsed_sec,
        "overrides_json": json.dumps(overrides, ensure_ascii=False, sort_keys=True),
    }


def _select_best_candidate(records: Sequence[Dict]) -> Dict:
    def _score(record: Dict) -> Tuple[float, int, int, float, float]:
        p_value = record["p_value"]
        if not np.isfinite(p_value):
            p_value = 1.0
        return (
            float(record["accuracy_mean"]),
            int(record["wins"]),
            -int(record["losses"]),
            float(record["delta_vs_ra"]),
            -float(p_value),
        )

    return max(records, key=_score)


def _plot_w_trajectories(
    detail_paths_by_method: Dict[str, Dict[int, Path]],
    focus_subjects: Sequence[int],
    out_dir: Path,
) -> None:
    for subject in focus_subjects:
        plt.figure(figsize=(12, 4))
        any_series = False
        for method_name, method_paths in detail_paths_by_method.items():
            detail_path = method_paths.get(subject)
            if detail_path is None:
                continue
            data = np.load(detail_path)
            if "w" not in data:
                continue
            plt.plot(data["w"], label=method_name, linewidth=1.5)
            any_series = True
        if not any_series:
            plt.close()
            continue
        plt.ylim(0.0, 1.0)
        plt.xlabel("Trial")
        plt.ylabel("Alignment weight w")
        plt.title(f"Weight trajectories (Target A{subject:02d})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"w_trajectory_targetA{subject:02d}.pdf", dpi=300)
        plt.close()


def _plot_window_accuracy_curves(
    detail_paths_by_method: Dict[str, Dict[int, Path]],
    focus_subjects: Sequence[int],
    out_dir: Path,
    window_size: int,
) -> None:
    for subject in focus_subjects:
        plt.figure(figsize=(12, 4))
        any_series = False
        for method_name, method_paths in detail_paths_by_method.items():
            detail_path = method_paths.get(subject)
            if detail_path is None:
                continue
            data = np.load(detail_path)
            if "correct" not in data:
                continue
            x_axis, window_acc = sliding_window_mean(data["correct"], window_size=window_size)
            if window_acc.size == 0:
                continue
            plt.plot(x_axis, window_acc, label=method_name, linewidth=1.5)
            any_series = True
        if not any_series:
            plt.close()
            continue
        plt.ylim(0.0, 1.0)
        plt.xlabel("Trial")
        plt.ylabel(f"Window accuracy (W={window_size})")
        plt.title(f"Windowed accuracy (Target A{subject:02d})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"window_accuracy_targetA{subject:02d}.pdf", dpi=300)
        plt.close()


def _plot_accuracy_vs_weight(
    detail_paths: Dict[int, Path],
    out_path: Path,
    window_size: int,
) -> None:
    weight_means, acc_means = [], []
    for detail_path in detail_paths.values():
        data = np.load(detail_path)
        if "w" not in data or "correct" not in data:
            continue
        _, w_mean = sliding_window_mean(data["w"], window_size=window_size)
        _, acc_mean = sliding_window_mean(data["correct"], window_size=window_size)
        min_n = min(len(w_mean), len(acc_mean))
        if min_n == 0:
            continue
        weight_means.extend(w_mean[:min_n].tolist())
        acc_means.extend(acc_mean[:min_n].tolist())

    if not weight_means:
        return

    plt.figure(figsize=(5, 5))
    plt.scatter(weight_means, acc_means, alpha=0.7, s=18)
    plt.xlabel(f"Mean weight w (window={window_size})")
    plt.ylabel("Window accuracy")
    plt.title("Accuracy vs weight")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_context_distribution(
    detail_paths: Dict[int, Path],
    out_path: Path,
) -> None:
    feature_arrays = []
    for detail_path in detail_paths.values():
        data = np.load(detail_path)
        if "context" not in data:
            continue
        feature_arrays.append(np.asarray(data["context"], dtype=np.float64))

    if not feature_arrays:
        return

    stacked = np.concatenate(feature_arrays, axis=0)
    plt.figure(figsize=(max(6, stacked.shape[1] * 1.6), 4))
    plt.boxplot(
        [stacked[:, idx] for idx in range(stacked.shape[1])],
        labels=[f"c{idx}" for idx in range(stacked.shape[1])],
    )
    plt.ylabel("Normalized context value")
    plt.title("Context feature distribution")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _write_negative_transfer_cards(
    subject_deltas: pd.DataFrame,
    best_detail_paths: Dict[int, Path],
    out_dir: Path,
    window_size: int,
) -> List[int]:
    negative_subjects: List[int] = []
    for row in subject_deltas.itertuples(index=False):
        if float(row.delta) >= 0.0:
            continue
        subject = int(row.target_subject)
        negative_subjects.append(subject)
        detail_path = best_detail_paths.get(subject)
        w_mean, w_std, worst_window_acc = float("nan"), float("nan"), float("nan")
        if detail_path is not None:
            data = np.load(detail_path)
            if "w" in data:
                w = np.asarray(data["w"], dtype=np.float64)
                w_mean = float(w.mean()) if w.size else float("nan")
                w_std = float(w.std()) if w.size else float("nan")
            if "correct" in data:
                _, window_acc = sliding_window_mean(data["correct"], window_size=window_size)
                if window_acc.size:
                    worst_window_acc = float(window_acc.min())

        card = "\n".join(
            [
                f"# Negative Transfer Case: Target A{subject:02d}",
                "",
                f"- RA accuracy: {float(row.accuracy_reference):.4f}",
                f"- Best adaptive accuracy: {float(row.accuracy_candidate):.4f}",
                f"- Delta vs RA: {float(row.delta):+.4f}",
                f"- Mean w: {w_mean:.4f}",
                f"- Std w: {w_std:.4f}",
                f"- Worst window accuracy: {worst_window_acc:.4f}",
            ]
        )
        (out_dir / f"negative_transfer_targetA{subject:02d}.md").write_text(
            card, encoding="utf-8"
        )
    return negative_subjects


def _gate_summary(
    base_summary: Dict,
    fixed_w1_summary: Dict,
    best_summary: Dict,
    base_runtime: float,
    ra_runtime: float,
) -> Dict[str, Dict[str, str]]:
    gate1_pass = fixed_w1_summary["mean"] > base_summary["mean"] and fixed_w1_summary["losses"] <= base_summary["losses"]
    gate2_pass = best_summary["delta_vs_reference"] > 0.0 and best_summary["wins"] > base_summary["wins"]
    runtime_ratio = float(base_runtime / ra_runtime) if ra_runtime > 0.0 else float("nan")
    gate3_pass = (
        best_summary["delta_vs_reference"] >= 0.03
        and best_summary["wins"] >= 5
        and best_summary["losses"] <= 4
        and (not np.isfinite(best_summary["p_value"]) or best_summary["p_value"] <= 0.1)
        and (not np.isfinite(runtime_ratio) or runtime_ratio <= 2.0)
    )
    return {
        "gate1": {
            "status": "PASS" if gate1_pass else "FAIL",
            "detail": "固定 w=1.0 是否持续压过当前 adaptive，用于判断是否切到保守 gating 主线。",
        },
        "gate2": {
            "status": "PASS" if gate2_pass else "FAIL",
            "detail": "best adaptive 是否已经平均优于 RA 且被试胜场数增加。",
        },
        "gate3": {
            "status": "PASS" if gate3_pass else "FAIL",
            "detail": "是否满足 3-5% 提升、至少 5/9 被试获益、统计方向支持与开销可解释。",
        },
    }


def _write_iteration_summary(
    path: Path,
    comparison_summary_df: pd.DataFrame,
    best_candidate: Dict,
    base_negative_subjects: Sequence[int],
    best_negative_subjects: Sequence[int],
    gates: Dict[str, Dict[str, str]],
    mechanism_figure: Path,
) -> None:
    header = "| method | accuracy_mean | accuracy_std | delta_vs_ra | wins | losses | p_value | effect_size | elapsed_sec |"
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, separator]
    for row in comparison_summary_df.itertuples(index=False):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.method),
                    f"{float(row.accuracy_mean):.4f}",
                    f"{float(row.accuracy_std):.4f}",
                    f"{float(row.delta_vs_ra):+.4f}",
                    str(int(row.wins)),
                    str(int(row.losses)),
                    f"{float(row.p_value):.4f}" if np.isfinite(row.p_value) else "nan",
                    f"{float(row.effect_size):.4f}" if np.isfinite(row.effect_size) else "nan",
                    f"{float(row.elapsed_sec):.2f}",
                ]
            )
            + " |"
        )
    summary_table = "\n".join(rows)

    lines = [
        "# Stage2 Rescue Iteration Summary",
        "",
        "## Final Comparison",
        summary_table,
        "",
        "## Selected Adaptive Candidate",
        f"- Candidate: `{best_candidate['candidate']}`",
        f"- Stage: `{best_candidate['stage']}`",
        f"- Overrides: `{best_candidate['overrides_json']}`",
        f"- Mechanism figure: `{mechanism_figure}`",
        "",
        "## Negative Transfer",
        f"- Base adaptive negative-transfer subjects: {list(base_negative_subjects)}",
        f"- Best adaptive negative-transfer subjects: {list(best_negative_subjects)}",
        "",
        "## Gates",
    ]
    for gate_name, gate_info in gates.items():
        lines.append(f"- {gate_name}: **{gate_info['status']}** — {gate_info['detail']}")

    if gates["gate3"]["status"] != "PASS":
        lines.extend(
            [
                "",
                "## Fallback Note",
                "- 当前结果尚未满足顶会主线门槛；建议把本轮输出直接转成 failure-analysis / pivot 备忘录输入 stage3。",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_detail_paths(primary: Dict[int, Path], extra: Dict[int, Path]) -> Dict[int, Path]:
    merged = dict(primary)
    merged.update(extra)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="latest")
    parser.add_argument("--targets", default=None, help="Comma separated target subjects")
    parser.add_argument(
        "--focus-subjects",
        default=None,
        help="Comma separated subjects for detailed diagnostics; default = subject 2 + base negatives",
    )
    parser.add_argument("--window-size", type=int, default=32)
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
    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/stage2/rescue/{args.run_name}")
    scan_dir = ensure_dir(str(Path(root_dir) / "scan"))
    figures_dir = ensure_dir(str(Path(root_dir) / "figures"))
    diagnostics_dir = ensure_dir(str(Path(root_dir) / "diagnostics"))
    details_dir = ensure_dir(str(Path(root_dir) / "details"))

    baselines: Dict[str, pd.DataFrame] = {}
    baseline_runtimes: Dict[str, float] = {}
    for method in ("noalign", "ea", "ra"):
        df, elapsed = _run_static_loso(
            loader=loader,
            all_subjects=all_subjects,
            target_subjects=target_subjects,
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            method=method,
            cov_eps=cov_eps,
        )
        baselines[method] = df
        baseline_runtimes[method] = elapsed
        df.to_csv(Path(scan_dir) / f"{method}_loso.csv", index=False)

    ra_df = baselines["ra"]

    base_df, base_elapsed, _ = _run_adaptive_loso(
        loader=loader,
        all_subjects=all_subjects,
        target_subjects=target_subjects,
        pre=pre,
        csp_kwargs=csp_kwargs,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        base_model_cfg=model_cfg,
        method_name="adaptive/base",
        dca_overrides=None,
        detail_subjects=None,
        details_dir=None,
    )
    base_df.to_csv(Path(scan_dir) / "adaptive_base_loso.csv", index=False)
    base_summary = summarize_against_reference(base_df, ra_df, metric="accuracy")
    base_subject_deltas = pair_subject_deltas(base_df, ra_df, metric="accuracy")
    base_subject_deltas.to_csv(Path(diagnostics_dir) / "base_adaptive_vs_ra_subject_deltas.csv", index=False)

    base_negative_subjects = base_subject_deltas.loc[base_subject_deltas["delta"] < 0.0, "target_subject"].astype(int).tolist()
    focus_subjects = _resolve_focus_subjects(args.focus_subjects, base_negative_subjects, target_subjects)

    candidate_records: List[Dict] = []
    scan_results: Dict[str, pd.DataFrame] = {"adaptive/base": base_df}

    def _evaluate_stage(stage: str, candidates: Sequence[Tuple[str, Dict]]) -> Dict:
        stage_records: List[Dict] = []
        for name, overrides in candidates:
            df, elapsed, _ = _run_adaptive_loso(
                loader=loader,
                all_subjects=all_subjects,
                target_subjects=target_subjects,
                pre=pre,
                csp_kwargs=csp_kwargs,
                lda_kwargs=lda_kwargs,
                cov_eps=cov_eps,
                base_model_cfg=model_cfg,
                method_name=name,
                dca_overrides=overrides,
                detail_subjects=None,
                details_dir=None,
            )
            scan_results[name] = df
            df.to_csv(Path(scan_dir) / f"{_sanitize_label(name)}_loso.csv", index=False)
            record = _candidate_record(stage, name, overrides, df, ra_df, elapsed)
            stage_records.append(record)
            candidate_records.append(record)

        stage_df = pd.DataFrame(stage_records).sort_values(
            ["accuracy_mean", "wins", "delta_vs_ra"], ascending=[False, False, False]
        )
        stage_df.to_csv(Path(scan_dir) / f"{stage}_summary.csv", index=False)
        return _select_best_candidate(stage_records)

    base_cond_cfg = model_cfg["dca_bgf"]["conditional"]
    base_ctx_cfg = model_cfg["dca_bgf"]["context"]
    base_weights = list(base_cond_cfg.get("weights", [1.2, 0.6, 0.3]))

    best_bias = _evaluate_stage(
        "bias_scan",
        [
            (f"adaptive/bias={bias:.2f}", {"conditional": {"bias": bias}})
            for bias in [float(base_cond_cfg.get("bias", 0.0)), 0.5, 1.0]
        ],
    )

    best_weight = _evaluate_stage(
        "weight_scale_scan",
        [
            (
                f"adaptive/scale={scale:.2f}",
                {
                    "conditional": {
                        "bias": json.loads(best_bias["overrides_json"]).get("conditional", {}).get("bias", float(base_cond_cfg.get("bias", 0.0))),
                        "weights": [float(weight) * scale for weight in base_weights],
                    }
                },
            )
            for scale in [1.0, 1.25, 1.5]
        ],
    )

    best_temp = _evaluate_stage(
        "temperature_ema_scan",
        [
            (
                f"adaptive/temp={temperature:.2f}_ema={ema:.2f}",
                _deep_merge(
                    json.loads(best_weight["overrides_json"]),
                    {"conditional": {"temperature": temperature, "ema_smooth_alpha": ema}},
                ),
            )
            for temperature in [1.0, 0.75]
            for ema in [0.0, 0.2, 0.4]
        ],
    )

    best_temp_overrides = json.loads(best_temp["overrides_json"])
    best_temp_weights = best_temp_overrides.get("conditional", {}).get("weights", base_weights)
    best_context = _evaluate_stage(
        "context_feature_scan",
        [
            (
                "adaptive/context=base",
                best_temp_overrides,
            ),
            (
                "adaptive/context=base+d_geo",
                _deep_merge(
                    best_temp_overrides,
                    {
                        "context": {
                            "features": ["d_src", "d_tgt", "sigma_recent", "d_geo"],
                            "cov_eps": float(base_ctx_cfg.get("cov_eps", 1.0e-6)),
                        },
                        "conditional": {
                            "weights": [*best_temp_weights, float(best_temp_weights[0])],
                        },
                    },
                ),
            ),
        ],
    )

    best_feedback = _evaluate_stage(
        "feedback_stability_scan",
        [
            (
                "adaptive/feedback=sum",
                json.loads(best_context["overrides_json"]),
            ),
            (
                "adaptive/feedback=entropy_priority",
                _deep_merge(
                    json.loads(best_context["overrides_json"]),
                    {"feedback": {"conflict_mode": "entropy_priority", "conf_trend_alpha": 0.15}},
                ),
            ),
            (
                "adaptive/feedback=average",
                _deep_merge(
                    json.loads(best_context["overrides_json"]),
                    {"feedback": {"conflict_mode": "average", "conf_trend_alpha": 0.10}},
                ),
            ),
        ],
    )

    fixed_w05_df, fixed_w05_elapsed, fixed_w05_details = _run_adaptive_loso(
        loader=loader,
        all_subjects=all_subjects,
        target_subjects=target_subjects,
        pre=pre,
        csp_kwargs=csp_kwargs,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        base_model_cfg=model_cfg,
        method_name="fixed/w=0.5",
        dca_overrides={"conditional": {"type": "fixed", "weight": 0.5}, "feedback": {"enabled": False}},
        detail_subjects=focus_subjects,
        details_dir=Path(details_dir),
    )
    fixed_w1_df, fixed_w1_elapsed, fixed_w1_details = _run_adaptive_loso(
        loader=loader,
        all_subjects=all_subjects,
        target_subjects=target_subjects,
        pre=pre,
        csp_kwargs=csp_kwargs,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        base_model_cfg=model_cfg,
        method_name="fixed/w=1.0",
        dca_overrides={"conditional": {"type": "fixed", "weight": 1.0}, "feedback": {"enabled": False}},
        detail_subjects=focus_subjects,
        details_dir=Path(details_dir),
    )
    base_detail_df, base_detail_elapsed, base_detail_paths = _run_adaptive_loso(
        loader=loader,
        all_subjects=all_subjects,
        target_subjects=target_subjects,
        pre=pre,
        csp_kwargs=csp_kwargs,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        base_model_cfg=model_cfg,
        method_name="adaptive/base",
        dca_overrides=None,
        detail_subjects=focus_subjects,
        details_dir=Path(details_dir),
    )
    best_df, best_elapsed, best_detail_paths = _run_adaptive_loso(
        loader=loader,
        all_subjects=all_subjects,
        target_subjects=target_subjects,
        pre=pre,
        csp_kwargs=csp_kwargs,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        base_model_cfg=model_cfg,
        method_name="adaptive/best",
        dca_overrides=json.loads(best_feedback["overrides_json"]),
        detail_subjects=focus_subjects,
        details_dir=Path(details_dir),
    )
    ra_detail_paths = _run_ra_detail_targets(
        loader=loader,
        all_subjects=all_subjects,
        targets=focus_subjects,
        pre=pre,
        csp_kwargs=csp_kwargs,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        details_dir=Path(details_dir),
    )
    best_subject_deltas = pair_subject_deltas(best_df, ra_df, metric="accuracy")
    best_subject_deltas.to_csv(Path(diagnostics_dir) / "best_adaptive_vs_ra_subject_deltas.csv", index=False)

    best_negative_subjects = (
        best_subject_deltas.loc[best_subject_deltas["delta"] < 0.0, "target_subject"]
        .astype(int)
        .tolist()
    )
    missing_focus_subjects = [subject for subject in best_negative_subjects if subject not in focus_subjects]
    if missing_focus_subjects:
        logger.info("Collecting extra details for late negative-transfer subjects: %s", missing_focus_subjects)
        focus_subjects = sorted(set(focus_subjects) | set(missing_focus_subjects))
        _df, _elapsed, extra_base_paths = _run_adaptive_loso(
            loader=loader,
            all_subjects=all_subjects,
            target_subjects=target_subjects,
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            cov_eps=cov_eps,
            base_model_cfg=model_cfg,
            method_name="adaptive/base",
            dca_overrides=None,
            detail_subjects=missing_focus_subjects,
            details_dir=Path(details_dir),
        )
        _df, _elapsed, extra_best_paths = _run_adaptive_loso(
            loader=loader,
            all_subjects=all_subjects,
            target_subjects=target_subjects,
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            cov_eps=cov_eps,
            base_model_cfg=model_cfg,
            method_name="adaptive/best",
            dca_overrides=json.loads(best_feedback["overrides_json"]),
            detail_subjects=missing_focus_subjects,
            details_dir=Path(details_dir),
        )
        _df, _elapsed, extra_fixed_w1_paths = _run_adaptive_loso(
            loader=loader,
            all_subjects=all_subjects,
            target_subjects=target_subjects,
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            cov_eps=cov_eps,
            base_model_cfg=model_cfg,
            method_name="fixed/w=1.0",
            dca_overrides={"conditional": {"type": "fixed", "weight": 1.0}, "feedback": {"enabled": False}},
            detail_subjects=missing_focus_subjects,
            details_dir=Path(details_dir),
        )
        extra_ra_paths = _run_ra_detail_targets(
            loader=loader,
            all_subjects=all_subjects,
            targets=missing_focus_subjects,
            pre=pre,
            csp_kwargs=csp_kwargs,
            lda_kwargs=lda_kwargs,
            cov_eps=cov_eps,
            details_dir=Path(details_dir),
        )
        base_detail_paths = _merge_detail_paths(base_detail_paths, extra_base_paths)
        best_detail_paths = _merge_detail_paths(best_detail_paths, extra_best_paths)
        fixed_w1_details = _merge_detail_paths(fixed_w1_details, extra_fixed_w1_paths)
        ra_detail_paths = _merge_detail_paths(ra_detail_paths, extra_ra_paths)

    comparison_methods = {
        "noalign": (baselines["noalign"], baseline_runtimes["noalign"]),
        "ea": (baselines["ea"], baseline_runtimes["ea"]),
        "ra": (baselines["ra"], baseline_runtimes["ra"]),
        "fixed/w=0.5": (fixed_w05_df, fixed_w05_elapsed),
        "fixed/w=1.0": (fixed_w1_df, fixed_w1_elapsed),
        "adaptive/best": (best_df, best_elapsed),
    }

    stacked_rows = []
    comparison_records = []
    for method_name, (df, elapsed_sec) in comparison_methods.items():
        local_df = df.copy()
        local_df["method"] = method_name
        stacked_rows.append(local_df)
        if method_name == "ra":
            summary = {
                "mean": float(local_df["accuracy"].mean()),
                "std": float(local_df["accuracy"].std(ddof=1)),
                "delta_vs_reference": 0.0,
                "wins": 0,
                "losses": 0,
                "draws": len(local_df),
                "p_value": float("nan"),
                "effect_size": float("nan"),
            }
        else:
            summary = summarize_against_reference(local_df, ra_df, metric="accuracy")
        comparison_records.append(
            {
                "method": method_name,
                "accuracy_mean": summary["mean"],
                "accuracy_std": summary["std"],
                "delta_vs_ra": summary["delta_vs_reference"],
                "wins": summary["wins"],
                "losses": summary["losses"],
                "draws": summary["draws"],
                "p_value": summary["p_value"],
                "effect_size": summary["effect_size"],
                "elapsed_sec": elapsed_sec,
            }
        )

    comparison_loso_df = pd.concat(stacked_rows, axis=0, ignore_index=True)
    comparison_summary_df = pd.DataFrame(comparison_records).sort_values(
        ["accuracy_mean", "wins"], ascending=[False, False]
    )
    comparison_loso_df.to_csv(Path(root_dir) / "comparison_loso.csv", index=False)
    comparison_summary_df.to_csv(Path(root_dir) / "comparison_summary.csv", index=False)

    detail_paths_by_method = {
        "ra": ra_detail_paths,
        "adaptive/base": base_detail_paths,
        "adaptive/best": best_detail_paths,
        "fixed/w=1.0": fixed_w1_details,
    }
    _plot_w_trajectories(detail_paths_by_method, focus_subjects, Path(figures_dir))
    _plot_window_accuracy_curves(
        detail_paths_by_method, focus_subjects, Path(figures_dir), window_size=args.window_size
    )
    mechanism_figure = Path(figures_dir) / "accuracy_vs_weight_best_adaptive.pdf"
    _plot_accuracy_vs_weight(best_detail_paths, mechanism_figure, window_size=args.window_size)
    _plot_context_distribution(best_detail_paths, Path(figures_dir) / "context_distribution_best_adaptive.pdf")

    base_negative_cards = _write_negative_transfer_cards(
        base_subject_deltas,
        base_detail_paths,
        Path(diagnostics_dir),
        window_size=args.window_size,
    )
    best_negative_cards = _write_negative_transfer_cards(
        best_subject_deltas,
        best_detail_paths,
        Path(diagnostics_dir),
        window_size=args.window_size,
    )

    fixed_w1_summary = summarize_against_reference(fixed_w1_df, ra_df, metric="accuracy")
    best_summary = summarize_against_reference(best_df, ra_df, metric="accuracy")
    gates = _gate_summary(
        base_summary=base_summary,
        fixed_w1_summary=fixed_w1_summary,
        best_summary=best_summary,
        base_runtime=best_elapsed,
        ra_runtime=baseline_runtimes["ra"],
    )
    _save_json(gates, Path(root_dir) / "gates.json")

    best_candidate_record = dict(best_feedback)
    _save_json(best_candidate_record, Path(root_dir) / "best_candidate.json")
    _write_iteration_summary(
        path=Path(root_dir) / "iteration_summary.md",
        comparison_summary_df=comparison_summary_df,
        best_candidate=best_candidate_record,
        base_negative_subjects=base_negative_cards,
        best_negative_subjects=best_negative_cards,
        gates=gates,
        mechanism_figure=mechanism_figure,
    )

    logger.info("Stage2 rescue outputs saved to %s", root_dir)


if __name__ == "__main__":
    main()
