#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.koopman_alignment import (
    KoopmanConservativeResidualAligner,
    KoopmanFeatureProjector,
)
from src.evaluation.metrics import cka, compute_metrics
from src.evaluation.rbid import compute_rbid_from_pairwise
from src.features.covariance import compute_covariances
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("ksda_exp_e2_diagnostic_refresh")

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

E1_ORIGINAL_METHOD = "Conservative Koopman aligner-r48"
E1_REFRESH_METHOD = "Conservative Koopman aligner-r48 (target-global refresh)"
E1_REFRESH_METHOD_KEY = "conservative-koopman-aligner-r48-target-global-refresh"
E1_ORIGINAL_REFRESHED_LABEL = "Conservative Koopman aligner-r48 (original pairwise)"
E1_ORIGINAL_REFRESHED_KEY = "conservative-koopman-aligner-r48-original-pairwise"
E2_METHOD = "RBID-aware Conservative Koopman aligner-r48"
PAIRWISE_PROTOCOL = "target-global pooled-source per target"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="2026-03-09-e2diag-r1")
    parser.add_argument("--e0-root", default="results/e0/2026-03-09-e0-r1")
    parser.add_argument("--e1-root", default="results/e1/2026-03-09-e1-r1")
    parser.add_argument("--e2-root", default="results/e2/2026-03-09-e2-r1")
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


def evaluate_conservative_pairwise_target_global(
    *,
    loader,
    subjects: Sequence[int],
    pre,
    rep_cfg: Mapping[str, object],
    aligner_cfg: Mapping[str, object],
    lda_kwargs: Mapping[str, object],
    cov_eps: float,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    n = len(subjects)
    acc = np.eye(n, dtype=np.float64)
    rep = np.eye(n, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    subject_to_idx = {int(subject): idx for idx, subject in enumerate(subjects)}

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
        psi_target_test_aligned = aligner.transform_target(psi_target_test)

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

            src_idx = subject_to_idx[int(source_subject)]
            tgt_idx = subject_to_idx[int(target_subject)]
            acc[src_idx, tgt_idx] = float(metrics["accuracy"])
            rep[src_idx, tgt_idx] = float(cka(psi_source_test_aligned, psi_target_test_aligned))
            rows.append(
                {
                    "source": int(source_subject),
                    "target": int(target_subject),
                    "accuracy": float(metrics["accuracy"]),
                    "kappa": float(metrics["kappa"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "cka": float(rep[src_idx, tgt_idx]),
                    "method": E1_REFRESH_METHOD,
                    "method_key": E1_REFRESH_METHOD_KEY,
                }
            )
            start = stop

    rbid = compute_rbid_from_pairwise(rep, acc)
    rep_values = rep[~np.eye(n, dtype=bool)]
    acc_values = acc[~np.eye(n, dtype=bool)]
    pearson_r = 0.0
    if not (np.allclose(rep_values, rep_values[0]) or np.allclose(acc_values, acc_values[0])):
        pearson_r = float(np.corrcoef(rep_values, acc_values)[0, 1])

    method_row = {
        "method": E1_REFRESH_METHOD,
        "method_key": E1_REFRESH_METHOD_KEY,
        "rbid": float(rbid["rbid"]),
        "rbid_pos": float(rbid["rbid_pos"]),
        "rbid_neg": float(rbid["rbid_neg"]),
        "tail_rbid": float(rbid["tail_rbid"]),
        "pearson_r": pearson_r,
        "pairwise_accuracy_mean": float(np.mean(acc_values)),
    }
    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True), method_row


def compute_target_rank_diagnostics(
    *,
    ra_pairwise: pd.DataFrame,
    method_pairwise: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    ra = ra_pairwise[["source", "target", "accuracy"]].rename(columns={"accuracy": "ra_accuracy"})
    rows: list[dict[str, float | int | str]] = []
    for method_name, pairwise_df in method_pairwise.items():
        merged = ra.merge(
            pairwise_df[["source", "target", "accuracy", "cka"]],
            on=["source", "target"],
            how="inner",
        )
        for target, group in merged.groupby("target"):
            rows.append(
                {
                    "target": int(target),
                    "method": str(method_name),
                    "corr_ra_vs_accuracy_spearman": float(
                        group["ra_accuracy"].corr(group["accuracy"], method="spearman")
                    ),
                    "corr_ra_vs_cka_spearman": float(
                        group["ra_accuracy"].corr(group["cka"], method="spearman")
                    ),
                    "corr_accuracy_vs_cka_spearman": float(
                        group["accuracy"].corr(group["cka"], method="spearman")
                    ),
                    "pairwise_accuracy_mean": float(group["accuracy"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "target"]).reset_index(drop=True)


def _augment_method_row(method_row: Mapping[str, object], pairwise_df: pd.DataFrame) -> Dict[str, object]:
    row = dict(method_row)
    row["pairwise_accuracy_mean"] = float(pairwise_df["accuracy"].mean())
    return row


def build_summary(
    *,
    run_dir: Path,
    e1_original_row: Mapping[str, object],
    refreshed_row: Mapping[str, object],
    e2_row: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "protocol_match": {
            "pairwise_protocol": PAIRWISE_PROTOCOL,
            "original_e1_pairwise_protocol": "per-source per-target refit",
        },
        "e1_original_pairwise": dict(e1_original_row),
        "e1_refreshed_pairwise": dict(refreshed_row),
        "e2_pairwise": dict(e2_row),
        "delta_refresh_vs_original": {
            "pairwise_accuracy_mean": float(refreshed_row["pairwise_accuracy_mean"]) - float(e1_original_row["pairwise_accuracy_mean"]),
            "rbid": float(refreshed_row["rbid"]) - float(e1_original_row["rbid"]),
            "tail_rbid": float(refreshed_row["tail_rbid"]) - float(e1_original_row["tail_rbid"]),
            "pearson_r": float(refreshed_row["pearson_r"]) - float(e1_original_row["pearson_r"]),
        },
        "delta_e2_vs_refresh": {
            "pairwise_accuracy_mean": float(e2_row["pairwise_accuracy_mean"]) - float(refreshed_row["pairwise_accuracy_mean"]),
            "rbid": float(e2_row["rbid"]) - float(refreshed_row["rbid"]),
            "tail_rbid": float(e2_row["tail_rbid"]) - float(refreshed_row["tail_rbid"]),
            "pearson_r": float(e2_row["pearson_r"]) - float(refreshed_row["pearson_r"]),
        },
        "diagnosis": {
            "e2_beats_refresh_on_rbid": bool(float(e2_row["rbid"]) < float(refreshed_row["rbid"])),
            "e2_beats_refresh_on_tail_rbid": bool(float(e2_row["tail_rbid"]) < float(refreshed_row["tail_rbid"])),
            "e2_beats_refresh_on_pairwise_accuracy": bool(
                float(e2_row["pairwise_accuracy_mean"]) >= float(refreshed_row["pairwise_accuracy_mean"])
            ),
        },
    }


def write_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, object],
    pairwise_panel: pd.DataFrame,
    rbid_panel: pd.DataFrame,
    diagnostics: pd.DataFrame,
    memo_text: str,
) -> None:
    ensure_dir(output_dir)
    pairwise_panel.to_csv(output_dir / "pairwise_scores.csv", index=False)
    rbid_panel.to_csv(output_dir / "rbid_method_comparison.csv", index=False)
    diagnostics.to_csv(output_dir / "target_rank_diagnostics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "memo.md").write_text(memo_text, encoding="utf-8")


def _render_memo(summary: Mapping[str, object], diagnostics: pd.DataFrame) -> str:
    diag_means = (
        diagnostics.groupby("method")[
            ["corr_ra_vs_accuracy_spearman", "corr_ra_vs_cka_spearman", "corr_accuracy_vs_cka_spearman"]
        ]
        .mean()
        .reset_index()
    )
    return (
        "# E2 Diagnostic Baseline Refresh Memo\n\n"
        f"**Run dir**: `{summary['run_dir']}`  \n"
        f"**Pairwise protocol**: `{summary['protocol_match']['pairwise_protocol']}`  \n"
        f"**Original E1 pairwise protocol**: `{summary['protocol_match']['original_e1_pairwise_protocol']}`\n\n"
        "## Key deltas\n\n"
        f"- Refresh vs original E1 RBID: `{summary['delta_refresh_vs_original']['rbid']:.4f}`\n"
        f"- Refresh vs original E1 Tail-RBID: `{summary['delta_refresh_vs_original']['tail_rbid']:.4f}`\n"
        f"- E2 vs refresh RBID: `{summary['delta_e2_vs_refresh']['rbid']:.4f}`\n"
        f"- E2 vs refresh Tail-RBID: `{summary['delta_e2_vs_refresh']['tail_rbid']:.4f}`\n"
        f"- E2 vs refresh pairwise accuracy mean: `{summary['delta_e2_vs_refresh']['pairwise_accuracy_mean']:.4f}`\n"
        f"- E2 vs refresh Pearson-r: `{summary['delta_e2_vs_refresh']['pearson_r']:.4f}`\n\n"
        "## Mean target-wise rank diagnostics\n\n"
        + diag_means.to_csv(index=False)
        + "\n## Verdict\n\n"
        + f"- E2 beats refreshed baseline on RBID: `{summary['diagnosis']['e2_beats_refresh_on_rbid']}`\n"
        + f"- E2 beats refreshed baseline on Tail-RBID: `{summary['diagnosis']['e2_beats_refresh_on_tail_rbid']}`\n"
        + f"- E2 beats refreshed baseline on pairwise accuracy: `{summary['diagnosis']['e2_beats_refresh_on_pairwise_accuracy']}`\n"
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
    e1_root = Path(args.e1_root)
    e2_root = Path(args.e2_root)
    run_dir = ensure_dir(Path("results/e2_diag") / args.run_name)

    start = time.perf_counter()
    refreshed_pairwise, refreshed_row = evaluate_conservative_pairwise_target_global(
        loader=loader,
        subjects=target_subjects,
        pre=pre,
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg=DEFAULT_ALIGNER_CFG,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
    )

    original_pairwise = pd.read_csv(e1_root / "pairwise_scores.csv")
    original_pairwise = original_pairwise.loc[original_pairwise["method"] == E1_ORIGINAL_METHOD].copy()
    original_pairwise["method"] = E1_ORIGINAL_REFRESHED_LABEL
    original_pairwise["method_key"] = E1_ORIGINAL_REFRESHED_KEY

    e2_pairwise = pd.read_csv(e2_root / "pairwise_scores.csv")
    e2_pairwise = e2_pairwise.loc[e2_pairwise["method"] == E2_METHOD].copy()

    e1_rbid = pd.read_csv(e1_root / "rbid_method_comparison.csv")
    e1_rbid = e1_rbid.loc[e1_rbid["method"] == E1_ORIGINAL_METHOD].iloc[0].to_dict()
    e2_rbid = pd.read_csv(e2_root / "rbid_method_comparison.csv")
    e2_rbid = e2_rbid.loc[e2_rbid["method"] == E2_METHOD].iloc[0].to_dict()

    original_row = _augment_method_row(e1_rbid, original_pairwise)
    original_row["method"] = E1_ORIGINAL_REFRESHED_LABEL
    original_row["method_key"] = E1_ORIGINAL_REFRESHED_KEY
    e2_row = _augment_method_row(e2_rbid, e2_pairwise)

    pairwise_panel = pd.concat([original_pairwise, refreshed_pairwise, e2_pairwise], ignore_index=True)
    pairwise_panel = pairwise_panel.sort_values(["method", "source", "target"]).reset_index(drop=True)
    rbid_panel = pd.DataFrame([original_row, refreshed_row, e2_row]).sort_values("rbid").reset_index(drop=True)

    e0_pairwise = pd.read_csv(e0_root / "phenomenon" / "r48" / "pairwise_scores.csv")
    ra_pairwise = e0_pairwise.loc[e0_pairwise["method"] == "RA"].copy()
    diagnostics = compute_target_rank_diagnostics(
        ra_pairwise=ra_pairwise,
        method_pairwise={
            E1_ORIGINAL_REFRESHED_LABEL: original_pairwise,
            E1_REFRESH_METHOD: refreshed_pairwise,
            E2_METHOD: e2_pairwise,
        },
    )

    summary = build_summary(
        run_dir=run_dir,
        e1_original_row=original_row,
        refreshed_row=refreshed_row,
        e2_row=e2_row,
    )
    summary["elapsed_sec"] = float(time.perf_counter() - start)

    memo_text = _render_memo(summary, diagnostics)
    write_outputs(
        output_dir=run_dir,
        summary=summary,
        pairwise_panel=pairwise_panel,
        rbid_panel=rbid_panel,
        diagnostics=diagnostics,
        memo_text=memo_text,
    )
    logger.info("Wrote E2 diagnostic refresh outputs to %s", run_dir)


if __name__ == "__main__":
    main()
