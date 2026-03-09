#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.koopman_alignment import (
    KoopmanConservativeResidualAligner,
    KoopmanFeatureProjector,
)
from src.evaluation.metrics import cka, compute_metrics
from src.evaluation.rbid import compute_rbid_from_pair_rows
from src.features.covariance import compute_covariances
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("ksda_exp_e2_proxy_diagnostic")

DEFAULT_REP_CFG = {"pca_rank": 48, "lifting": "quadratic"}
DEFAULT_ALIGNER_CFG = {
    "basis_method": "A1",
    "basis_k": 32,
    "basis_reg_lambda": 1.0e-4,
    "residual_rank": 8,
    "lambda_cls": 1.0,
    "lambda_dyn": 1.0,
    "lambda_rank": 0.0,
    "lambda_reg": 1.0e-3,
    "ridge_alpha": 1.0e-3,
    "max_iter": 100,
}

RUN_METHOD = "Conservative Koopman aligner-r48 (proxy-only diagnostic)"
RUN_METHOD_KEY = "conservative-koopman-aligner-r48-proxy-only-diagnostic"
CURRENT_TRAINING_PROXY = "proxy_train_mean_cosine"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="2026-03-09-e2proxy-r1")
    parser.add_argument("--e0-root", default="results/e0/2026-03-09-e0-r1")
    parser.add_argument("--refresh-root", default="results/e2_diag/2026-03-09-e2diag-r1")
    parser.add_argument("--targets", default=None)
    return parser


def _resolve_targets(all_subjects: Sequence[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def _make_aligner(aligner_cfg: Mapping[str, object]) -> KoopmanConservativeResidualAligner:
    return KoopmanConservativeResidualAligner(
        residual_rank=int(aligner_cfg["residual_rank"]),
        basis_k=int(aligner_cfg["basis_k"]),
        basis_reg_lambda=float(aligner_cfg["basis_reg_lambda"]),
        lambda_cls=float(aligner_cfg["lambda_cls"]),
        lambda_dyn=float(aligner_cfg["lambda_dyn"]),
        lambda_rank=float(aligner_cfg["lambda_rank"]),
        lambda_reg=float(aligner_cfg["lambda_reg"]),
        ridge_alpha=float(aligner_cfg["ridge_alpha"]),
        max_iter=int(aligner_cfg["max_iter"]),
    )


def _load_ra_prior(e0_root: Path) -> Dict[int, Dict[int, float]]:
    pairwise = pd.read_csv(e0_root / "phenomenon" / "r48" / "pairwise_scores.csv")
    ra_rows = pairwise.loc[pairwise["method"] == "RA"].copy()
    prior: Dict[int, Dict[int, float]] = {}
    for row in ra_rows.itertuples(index=False):
        prior.setdefault(int(row.target), {})[int(row.source)] = float(row.accuracy)
    return prior


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _negative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(-np.linalg.norm(a - b))


def _nan_to_zero(value: float) -> float:
    return 0.0 if pd.isna(value) else float(value)


def get_proxy_columns() -> list[str]:
    return [
        "proxy_train_mean_cosine",
        "proxy_test_mean_cosine",
        "proxy_train_mean_neg_l2",
        "proxy_test_mean_neg_l2",
        "proxy_test_cka",
    ]


def evaluate_proxy_only_pairwise(
    *,
    loader,
    subjects: Sequence[int],
    pre,
    rep_cfg: Mapping[str, object],
    aligner_cfg: Mapping[str, object],
    lda_kwargs: Mapping[str, object],
    cov_eps: float,
    ra_prior: Mapping[int, Mapping[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for target_subject in subjects:
        source_subject_ids = [int(subject) for subject in subjects if int(subject) != int(target_subject)]
        X_source_blocks = []
        y_source_blocks = []
        X_source_test_blocks = []
        for source_subject in source_subject_ids:
            X_source_train_subject, y_source_train_subject = loader.load_subject(source_subject, split="train")
            X_source_test_subject, _ = loader.load_subject(source_subject, split="test")
            X_source_blocks.append(X_source_train_subject)
            y_source_blocks.append(y_source_train_subject)
            X_source_test_blocks.append(X_source_test_subject)

        X_source_train = np.concatenate(X_source_blocks, axis=0)
        y_source_train = np.concatenate(y_source_blocks, axis=0)
        X_target_train, _ = loader.load_subject(int(target_subject), split="train")
        X_target_test, y_target_test = loader.load_subject(int(target_subject), split="test")

        X_source_train = pre.fit(X_source_train, y_source_train).transform(X_source_train)
        X_target_train = pre.transform(X_target_train)
        X_target_test = pre.transform(X_target_test)
        X_source_test_blocks = [pre.transform(block) for block in X_source_test_blocks]

        cov_source_train = compute_covariances(X_source_train, eps=cov_eps)
        cov_target_train = compute_covariances(X_target_train, eps=cov_eps)
        cov_target_test = compute_covariances(X_target_test, eps=cov_eps)

        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(cov_source_train)
        psi_source_train = projector.transform(cov_source_train)
        psi_target_train = projector.transform(cov_target_train)
        psi_target_test = projector.transform(cov_target_test)

        aligner = _make_aligner(aligner_cfg).fit(
            psi_source_train,
            psi_target_train,
            y_source=y_source_train,
            source_block_lengths=[len(block) for block in X_source_blocks],
        )
        psi_source_train_aligned = aligner.transform_source(psi_source_train)
        psi_target_train_aligned = aligner.transform_target(psi_target_train)
        psi_target_test_aligned = aligner.transform_target(psi_target_test)

        target_train_mean = psi_target_train_aligned.mean(axis=0)
        target_test_mean = psi_target_test_aligned.mean(axis=0)

        start = 0
        for source_subject, X_source_train_block, y_source_train_block, X_source_test_block in zip(
            source_subject_ids, X_source_blocks, y_source_blocks, X_source_test_blocks
        ):
            block_len = len(X_source_train_block)
            stop = start + block_len
            psi_source_train_block_aligned = psi_source_train_aligned[start:stop]

            cov_source_test = compute_covariances(X_source_test_block, eps=cov_eps)
            psi_source_test = projector.transform(cov_source_test)
            psi_source_test_aligned = aligner.transform_source(psi_source_test)

            lda = LDA(**lda_kwargs).fit(psi_source_train_block_aligned, y_source_train_block)
            y_pred = lda.predict(psi_target_test_aligned)
            metrics = compute_metrics(y_target_test, y_pred)

            source_train_mean = psi_source_train_block_aligned.mean(axis=0)
            source_test_mean = psi_source_test_aligned.mean(axis=0)
            rows.append(
                {
                    "source": int(source_subject),
                    "target": int(target_subject),
                    "accuracy": float(metrics["accuracy"]),
                    "kappa": float(metrics["kappa"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "ra_accuracy": float(ra_prior[int(target_subject)][int(source_subject)]),
                    "proxy_train_mean_cosine": _safe_cosine(source_train_mean, target_train_mean),
                    "proxy_test_mean_cosine": _safe_cosine(source_test_mean, target_test_mean),
                    "proxy_train_mean_neg_l2": _negative_l2(source_train_mean, target_train_mean),
                    "proxy_test_mean_neg_l2": _negative_l2(source_test_mean, target_test_mean),
                    "proxy_test_cka": float(cka(psi_source_test_aligned, psi_target_test_aligned)),
                    "method": RUN_METHOD,
                    "method_key": RUN_METHOD_KEY,
                }
            )
            start = stop

    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True)


def summarize_proxy_scores(
    *,
    pairwise_df: pd.DataFrame,
    proxy_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proxy_rows: list[dict[str, float | str]] = []
    target_rows: list[dict[str, float | int | str]] = []

    for proxy in proxy_columns:
        behavior_diag = compute_rbid_from_pair_rows(pairwise_df, rep_col=proxy, beh_col="accuracy")
        ra_diag = compute_rbid_from_pair_rows(pairwise_df, rep_col=proxy, beh_col="ra_accuracy")
        global_behavior_pearson = _nan_to_zero(pairwise_df[proxy].corr(pairwise_df["accuracy"], method="pearson"))
        global_behavior_spearman = _nan_to_zero(pairwise_df[proxy].corr(pairwise_df["accuracy"], method="spearman"))
        global_ra_pearson = _nan_to_zero(pairwise_df[proxy].corr(pairwise_df["ra_accuracy"], method="pearson"))
        global_ra_spearman = _nan_to_zero(pairwise_df[proxy].corr(pairwise_df["ra_accuracy"], method="spearman"))

        per_target_behavior = []
        per_target_ra = []
        per_target_proxy_acc = []
        for target, group in pairwise_df.groupby("target"):
            corr_behavior = _nan_to_zero(group[proxy].corr(group["accuracy"], method="spearman"))
            corr_ra = _nan_to_zero(group[proxy].corr(group["ra_accuracy"], method="spearman"))
            corr_proxy_acc = _nan_to_zero(group[proxy].corr(group["accuracy"], method="pearson"))
            per_target_behavior.append(corr_behavior)
            per_target_ra.append(corr_ra)
            per_target_proxy_acc.append(corr_proxy_acc)
            target_rows.append(
                {
                    "target": int(target),
                    "proxy": str(proxy),
                    "corr_behavior_spearman": corr_behavior,
                    "corr_ra_spearman": corr_ra,
                    "corr_behavior_pearson": corr_proxy_acc,
                    "pairwise_accuracy_mean": float(group["accuracy"].mean()),
                    "pairwise_ra_accuracy_mean": float(group["ra_accuracy"].mean()),
                }
            )

        proxy_rows.append(
            {
                "proxy": str(proxy),
                "rbid_vs_behavior": float(behavior_diag["rbid"]),
                "tail_rbid_vs_behavior": float(behavior_diag["tail_rbid"]),
                "pearson_vs_behavior": global_behavior_pearson,
                "spearman_vs_behavior": global_behavior_spearman,
                "mean_target_corr_behavior_spearman": float(np.mean(per_target_behavior)),
                "mean_target_corr_behavior_pearson": float(np.mean(per_target_proxy_acc)),
                "rbid_vs_ra": float(ra_diag["rbid"]),
                "tail_rbid_vs_ra": float(ra_diag["tail_rbid"]),
                "pearson_vs_ra": global_ra_pearson,
                "spearman_vs_ra": global_ra_spearman,
                "mean_target_corr_ra_spearman": float(np.mean(per_target_ra)),
            }
        )

    proxy_summary = pd.DataFrame(proxy_rows).sort_values(
        ["rbid_vs_behavior", "tail_rbid_vs_behavior", "mean_target_corr_behavior_spearman"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    target_summary = pd.DataFrame(target_rows).sort_values(["proxy", "target"]).reset_index(drop=True)
    return proxy_summary, target_summary


def build_summary(*, run_dir: Path, proxy_summary: pd.DataFrame) -> Dict[str, object]:
    summary_frame = proxy_summary.copy()
    for column, default in [
        ("tail_rbid_vs_behavior", np.inf),
        ("mean_target_corr_behavior_spearman", -np.inf),
        ("tail_rbid_vs_ra", np.inf),
        ("mean_target_corr_ra_spearman", -np.inf),
    ]:
        if column not in summary_frame.columns:
            summary_frame[column] = default

    best_behavior = summary_frame.sort_values(
        ["rbid_vs_behavior", "tail_rbid_vs_behavior", "mean_target_corr_behavior_spearman"],
        ascending=[True, True, False],
    ).iloc[0].to_dict()
    best_ra = summary_frame.sort_values(
        ["rbid_vs_ra", "tail_rbid_vs_ra", "mean_target_corr_ra_spearman"],
        ascending=[True, True, False],
    ).iloc[0].to_dict()
    current = summary_frame.loc[summary_frame["proxy"] == CURRENT_TRAINING_PROXY]
    current_row = current.iloc[0].to_dict() if not current.empty else {"proxy": CURRENT_TRAINING_PROXY}
    return {
        "run_dir": str(run_dir),
        "current_training_proxy": current_row,
        "best_proxy_by_behavior": best_behavior,
        "best_proxy_by_ra": best_ra,
    }


def write_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, object],
    pairwise_df: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    memo_text: str,
) -> None:
    ensure_dir(output_dir)
    pairwise_df.to_csv(output_dir / "pairwise_proxy_scores.csv", index=False)
    proxy_summary.to_csv(output_dir / "proxy_summary.csv", index=False)
    target_summary.to_csv(output_dir / "target_proxy_diagnostics.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "memo.md").write_text(memo_text, encoding="utf-8")


def _render_memo(summary: Mapping[str, object], proxy_summary: pd.DataFrame) -> str:
    cols = [
        "proxy",
        "rbid_vs_behavior",
        "tail_rbid_vs_behavior",
        "mean_target_corr_behavior_spearman",
        "rbid_vs_ra",
        "mean_target_corr_ra_spearman",
    ]
    return (
        "# E2 Proxy-only Diagnostic Memo\n\n"
        f"**Run dir**: `{summary['run_dir']}`\n\n"
        f"**Current training proxy**: `{summary['current_training_proxy']['proxy']}`\n"
        f"**Best proxy by behavior**: `{summary['best_proxy_by_behavior']['proxy']}`\n"
        f"**Best proxy by RA prior**: `{summary['best_proxy_by_ra']['proxy']}`\n\n"
        "## Proxy summary\n\n"
        + proxy_summary[cols].to_csv(index=False)
    )


def main(argv: list[str] | None = None) -> None:
    from src.data.loader import BCIDataLoader
    from src.data.preprocessing import Preprocessor

    args = build_argparser().parse_args(argv)
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
    lda_kwargs = model_cfg["lda"]

    e0_root = Path(args.e0_root)
    refresh_root = Path(args.refresh_root)
    run_dir = ensure_dir(Path("results/e2_proxy_diag") / args.run_name)

    start = time.perf_counter()
    ra_prior = _load_ra_prior(e0_root)
    pairwise_df = evaluate_proxy_only_pairwise(
        loader=loader,
        subjects=target_subjects,
        pre=pre,
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg=DEFAULT_ALIGNER_CFG,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        ra_prior=ra_prior,
    )
    proxy_summary, target_summary = summarize_proxy_scores(
        pairwise_df=pairwise_df,
        proxy_columns=get_proxy_columns(),
    )
    summary = build_summary(run_dir=run_dir, proxy_summary=proxy_summary)

    if (refresh_root / "pairwise_scores.csv").exists():
        refresh_pairwise = pd.read_csv(refresh_root / "pairwise_scores.csv")
        refresh_rows = refresh_pairwise.loc[
            refresh_pairwise["method"] == "Conservative Koopman aligner-r48 (target-global refresh)"
        ][["source", "target", "accuracy", "cka"]].copy()
        merged = refresh_rows.merge(
            pairwise_df[["source", "target", "accuracy", "proxy_test_cka"]],
            on=["source", "target"],
            how="inner",
        )
        if not merged.empty:
            summary["refresh_consistency"] = {
                "n_pairs": int(len(merged)),
                "max_abs_accuracy_delta": float(np.max(np.abs(merged["accuracy_x"] - merged["accuracy_y"]))),
                "max_abs_cka_delta": float(np.max(np.abs(merged["cka"] - merged["proxy_test_cka"]))),
            }

    summary["elapsed_sec"] = float(time.perf_counter() - start)
    memo_text = _render_memo(summary, proxy_summary)
    write_outputs(
        output_dir=run_dir,
        summary=summary,
        pairwise_df=pairwise_df,
        proxy_summary=proxy_summary,
        target_summary=target_summary,
        memo_text=memo_text,
    )
    logger.info("Wrote E2 proxy-only diagnostic outputs to %s", run_dir)


if __name__ == "__main__":
    main()
