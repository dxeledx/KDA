#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

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


logger = get_logger("ksda_exp_e2a_proxy_replacement")

DEFAULT_REP_CFG = {"pca_rank": 48, "lifting": "quadratic"}
DEFAULT_ALIGNER_CFG = {
    "basis_method": "A1",
    "basis_k": 32,
    "basis_reg_lambda": 1.0e-4,
    "residual_rank": 8,
    "lambda_cls": 1.0,
    "lambda_dyn": 1.0,
    "lambda_rank": 1.0,
    "rank_score_mode": "mean_dyn_neg_l2",
    "rank_mean_weight": 1.0,
    "rank_dyn_weight": 1.0,
    "lambda_reg": 1.0e-3,
    "ridge_alpha": 1.0e-3,
    "max_iter": 100,
}

METHOD_NAME = "RBID-aware Conservative Koopman aligner-r48 (mean+dyn proxy)"
METHOD_KEY = "rbid-aware-conservative-koopman-aligner-r48-mean-dyn-proxy"
E1_CONTROL_METHOD = "Conservative Koopman aligner-r48"
E2_CONTROL_METHOD = "RBID-aware Conservative Koopman aligner-r48"
REFRESH_CONTROL_METHOD = "Conservative Koopman aligner-r48 (target-global refresh)"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="2026-03-09-e2a-r1")
    parser.add_argument("--e0-root", default="results/e0/2026-03-09-e0-r1")
    parser.add_argument("--e1-root", default="results/e1/2026-03-09-e1-r1")
    parser.add_argument("--e2-root", default="results/e2/2026-03-09-e2-r1")
    parser.add_argument("--refresh-root", default="results/e2_diag/2026-03-09-e2diag-r1")
    parser.add_argument("--targets", default=None)
    return parser


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
) -> List[Dict[str, np.ndarray | int | list[int]]]:
    folds = []
    for target_subject in target_subjects:
        X_source_blocks, y_source_blocks, source_subject_ids = [], [], []
        for subject in all_subjects:
            if subject == target_subject:
                continue
            X_train_subject, y_train_subject = loader.load_subject(int(subject), split="train")
            X_source_blocks.append(X_train_subject)
            y_source_blocks.append(y_train_subject)
            source_subject_ids.append(int(subject))

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
                "source_subject_ids": source_subject_ids,
            }
        )
    return folds


def _load_behavior_prior(e0_root: Path) -> Dict[int, Dict[int, float]]:
    pairwise = pd.read_csv(e0_root / "phenomenon" / "r48" / "pairwise_scores.csv")
    ra_rows = pairwise.loc[pairwise["method"] == "RA"].copy()
    prior: Dict[int, Dict[int, float]] = {}
    for row in ra_rows.itertuples(index=False):
        prior.setdefault(int(row.target), {})[int(row.source)] = float(row.accuracy)
    return prior


def _make_aligner(aligner_cfg: Dict[str, object]) -> KoopmanConservativeResidualAligner:
    return KoopmanConservativeResidualAligner(
        residual_rank=int(aligner_cfg["residual_rank"]),
        basis_k=int(aligner_cfg["basis_k"]),
        basis_reg_lambda=float(aligner_cfg["basis_reg_lambda"]),
        lambda_cls=float(aligner_cfg["lambda_cls"]),
        lambda_dyn=float(aligner_cfg["lambda_dyn"]),
        lambda_rank=float(aligner_cfg["lambda_rank"]),
        rank_score_mode=str(aligner_cfg["rank_score_mode"]),
        rank_mean_weight=float(aligner_cfg["rank_mean_weight"]),
        rank_dyn_weight=float(aligner_cfg["rank_dyn_weight"]),
        lambda_reg=float(aligner_cfg["lambda_reg"]),
        ridge_alpha=float(aligner_cfg["ridge_alpha"]),
        max_iter=int(aligner_cfg["max_iter"]),
    )


def evaluate_e2a_loso(
    *,
    folds: Sequence[Dict[str, np.ndarray | int | list[int]]],
    rep_cfg: Dict[str, object],
    aligner_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
    cov_eps: float,
    behavior_prior: Dict[int, Dict[int, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rows = []
    pair_rows = []
    for fold in folds:
        target_subject = int(fold["target_subject"])
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(fold["cov_source"])
        psi_source = projector.transform(fold["cov_source"])
        psi_target_train = projector.transform(fold["cov_target_train"])
        psi_target_test = projector.transform(fold["cov_target_test"])

        aligner = _make_aligner(aligner_cfg).fit(
            psi_source,
            psi_target_train,
            y_source=fold["y_source"],
            source_subject_ids=fold["source_subject_ids"],
            source_block_lengths=fold["source_block_lengths"],
            behavior_prior_scores=behavior_prior[target_subject],
        )
        psi_source_aligned = aligner.transform_source(psi_source)
        psi_target_test_aligned = aligner.transform_target(psi_target_test)

        lda = LDA(**lda_kwargs).fit(psi_source_aligned, fold["y_source"])
        y_pred = lda.predict(psi_target_test_aligned)
        metrics = compute_metrics(fold["y_target_test"], y_pred)
        rows.append(
            {
                "target_subject": target_subject,
                **metrics,
                "optimizer_success": bool(aligner.optimization_info_["success"]),
                "optimizer_nit": int(aligner.optimization_info_["nit"]),
                "loss_final": float(aligner.optimization_info_.get("fun", 0.0)),
            }
        )
        pair_rows.append(
            {
                "source": 0,
                "target": target_subject,
                "accuracy": float(metrics["accuracy"]),
                "kappa": float(metrics["kappa"]),
                "f1_macro": float(metrics["f1_macro"]),
                "cka": float(
                    cka(
                        psi_source_aligned[: len(psi_target_test_aligned)],
                        psi_target_test_aligned[: len(psi_source_aligned[: len(psi_target_test_aligned)])],
                    )
                ),
                "method": METHOD_NAME,
                "method_key": METHOD_KEY,
            }
        )

    loso_df = pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)
    pairwise_df = pd.DataFrame(pair_rows).sort_values("target").reset_index(drop=True)
    method_row = {
        "method": METHOD_NAME,
        "method_key": METHOD_KEY,
    }
    return loso_df, pairwise_df, method_row


def evaluate_e2a_pairwise(
    *,
    loader,
    subjects: Sequence[int],
    pre,
    rep_cfg: Dict[str, object],
    aligner_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
    cov_eps: float,
    behavior_prior: Dict[int, Dict[int, float]],
) -> tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    n = len(subjects)
    acc = np.eye(n, dtype=np.float64)
    rep = np.eye(n, dtype=np.float64)
    rows = []
    component_rows = []

    subject_to_idx = {int(subject): idx for idx, subject in enumerate(subjects)}
    for tgt in subjects:
        target_subject = int(tgt)
        source_subject_ids = [int(subject) for subject in subjects if int(subject) != target_subject]
        X_source_blocks, y_source_blocks, X_source_test_blocks = [], [], []
        for subject in source_subject_ids:
            X_source_train_subject, y_source_train_subject = loader.load_subject(subject, split="train")
            X_source_test_subject, _ = loader.load_subject(subject, split="test")
            X_source_blocks.append(X_source_train_subject)
            y_source_blocks.append(y_source_train_subject)
            X_source_test_blocks.append(X_source_test_subject)

        X_source_train = np.concatenate(X_source_blocks, axis=0)
        y_source_train = np.concatenate(y_source_blocks, axis=0)
        X_target_train, _ = loader.load_subject(target_subject, split="train")
        X_target_test, y_target_test = loader.load_subject(target_subject, split="test")

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

        block_lengths = [len(block) for block in X_source_blocks]
        aligner = _make_aligner(aligner_cfg).fit(
            psi_source_train,
            psi_target_train,
            y_source=y_source_train,
            source_subject_ids=source_subject_ids,
            source_block_lengths=block_lengths,
            behavior_prior_scores=behavior_prior[target_subject],
        )
        psi_source_train_aligned = aligner.transform_source(psi_source_train)
        psi_target_test_aligned = aligner.transform_target(psi_target_test)
        score_components = aligner.compute_rank_score_components(
            source_features=psi_source_train,
            target_features=psi_target_train,
            source_block_lengths=block_lengths,
        )
        score_components["target"] = int(target_subject)
        score_components["source"] = source_subject_ids
        score_components["behavior_prior"] = [float(behavior_prior[target_subject][source]) for source in source_subject_ids]

        start = 0
        for row_idx, (source_subject, X_source_train_block, y_source_train_block, X_source_test_block) in enumerate(
            zip(source_subject_ids, X_source_blocks, y_source_blocks, X_source_test_blocks)
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

            src_idx = subject_to_idx[source_subject]
            tgt_idx = subject_to_idx[target_subject]
            acc[src_idx, tgt_idx] = metrics["accuracy"]
            rep[src_idx, tgt_idx] = cka(psi_source_test_aligned, psi_target_test_aligned)
            rows.append(
                {
                    "source": int(source_subject),
                    "target": int(target_subject),
                    "accuracy": float(metrics["accuracy"]),
                    "kappa": float(metrics["kappa"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "cka": float(rep[src_idx, tgt_idx]),
                    "method": METHOD_NAME,
                    "method_key": METHOD_KEY,
                }
            )
            component_rows.append(
                {
                    "source": int(source_subject),
                    "target": int(target_subject),
                    "mean_sq_dist": float(score_components.loc[row_idx, "mean_sq_dist"]),
                    "dyn_resid": float(score_components.loc[row_idx, "dyn_resid"]),
                    "u_score": float(score_components.loc[row_idx, "u_score"]),
                    "behavior_prior": float(score_components.loc[row_idx, "behavior_prior"]),
                    "accuracy": float(metrics["accuracy"]),
                }
            )
            start = stop

    rbid = compute_rbid_from_pairwise(rep, acc)
    method_row = {
        "method": METHOD_NAME,
        "method_key": METHOD_KEY,
        "rbid": float(rbid["rbid"]),
        "rbid_pos": float(rbid["rbid_pos"]),
        "rbid_neg": float(rbid["rbid_neg"]),
        "tail_rbid": float(rbid["tail_rbid"]),
        "pearson_r": float(np.corrcoef(rep[~np.eye(n, dtype=bool)], acc[~np.eye(n, dtype=bool)])[0, 1]),
        "pairwise_accuracy_mean": float(acc[~np.eye(n, dtype=bool)].mean()),
    }
    return (
        pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True),
        method_row,
        pd.DataFrame(component_rows).sort_values(["source", "target"]).reset_index(drop=True),
    )


def build_summary(
    *,
    loso_df: pd.DataFrame,
    method_row: Dict[str, object],
    e1_summary: Dict[str, object],
    e2_summary: Dict[str, object],
    refresh_summary: Dict[str, object],
    run_dir: Path,
) -> Dict[str, object]:
    e1_control = dict(e1_summary["conservative"])
    e2_control = dict(e2_summary["rbid_aware"])
    refresh_control = dict(refresh_summary["e1_refreshed_pairwise"])
    e2_pairwise_accuracy_mean = float(e2_control.get("pairwise_accuracy_mean", e2_control.get("accuracy_mean", 0.0)))
    accuracy_mean = float(loso_df["accuracy"].mean())
    kappa_mean = float(loso_df["kappa"].mean())
    rbid = float(method_row["rbid"])
    tail_rbid = float(method_row["tail_rbid"])
    pairwise_accuracy_mean = float(method_row["pairwise_accuracy_mean"])
    return {
        "run_dir": str(run_dir),
        "representation": dict(DEFAULT_REP_CFG),
        "rank_score_mode": str(DEFAULT_ALIGNER_CFG["rank_score_mode"]),
        "rank_mean_weight": float(DEFAULT_ALIGNER_CFG["rank_mean_weight"]),
        "rank_dyn_weight": float(DEFAULT_ALIGNER_CFG["rank_dyn_weight"]),
        "control_loso_e1": e1_control,
        "control_pairwise_refresh": refresh_control,
        "secondary_control_e2": {**e2_control, "pairwise_accuracy_mean": e2_pairwise_accuracy_mean},
        "e2a_proxy_replacement": {
            "method": METHOD_NAME,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": float(loso_df["accuracy"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
            "kappa_mean": kappa_mean,
            "kappa_std": float(loso_df["kappa"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
            "rbid": rbid,
            "rbid_pos": float(method_row.get("rbid_pos", 0.0)),
            "rbid_neg": float(method_row.get("rbid_neg", 0.0)),
            "tail_rbid": tail_rbid,
            "pearson_r": float(method_row["pearson_r"]),
            "pairwise_accuracy_mean": pairwise_accuracy_mean,
        },
        "delta_vs_e1_loso": {
            "accuracy_mean": accuracy_mean - float(e1_control["accuracy_mean"]),
            "kappa_mean": kappa_mean - float(e1_control["kappa_mean"]),
        },
        "delta_vs_refresh_pairwise": {
            "rbid": rbid - float(refresh_control["rbid"]),
            "tail_rbid": tail_rbid - float(refresh_control["tail_rbid"]),
            "pearson_r": float(method_row["pearson_r"]) - float(refresh_control["pearson_r"]),
            "pairwise_accuracy_mean": pairwise_accuracy_mean - float(refresh_control["pairwise_accuracy_mean"]),
        },
        "delta_vs_e2_pairwise": {
            "rbid": rbid - float(e2_control["rbid"]),
            "tail_rbid": tail_rbid - float(e2_control["tail_rbid"]),
            "pearson_r": float(method_row["pearson_r"]) - float(e2_control["pearson_r"]),
            "pairwise_accuracy_mean": pairwise_accuracy_mean - e2_pairwise_accuracy_mean,
        },
        "gate": {
            "accuracy_pass": bool(accuracy_mean >= float(e1_control["accuracy_mean"])),
            "rbid_pass": bool(rbid < float(refresh_control["rbid"])),
            "tail_rbid_pass": bool(tail_rbid < float(refresh_control["tail_rbid"])),
            "ready_for_e2b": bool(
                accuracy_mean >= float(e1_control["accuracy_mean"])
                and rbid < float(refresh_control["rbid"])
                and tail_rbid < float(refresh_control["tail_rbid"])
            ),
        },
    }


def write_outputs(
    *,
    output_dir: Path,
    loso_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    rbid_df: pd.DataFrame,
    rank_components_df: pd.DataFrame,
    summary: Dict[str, object],
    memo_text: str,
) -> None:
    ensure_dir(output_dir)
    loso_df.to_csv(output_dir / "loso_subject_results.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_scores.csv", index=False)
    rbid_df.to_csv(output_dir / "rbid_method_comparison.csv", index=False)
    rank_components_df.to_csv(output_dir / "rank_score_components.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "memo.md").write_text(memo_text, encoding="utf-8")


def _build_pairwise_panel(
    e0_root: Path,
    e2_root: Path,
    refresh_root: Path,
    e2a_pairwise: pd.DataFrame,
) -> pd.DataFrame:
    e0_pairwise = pd.read_csv(e0_root / "phenomenon" / "r48" / "pairwise_scores.csv")
    e0_keep = e0_pairwise.loc[e0_pairwise["method"].isin(["RA", "Static Koopman aligner"])].copy()
    e0_keep["method"] = e0_keep["method"].replace({"Static Koopman aligner": "Static Koopman aligner-r48"})
    e0_keep["method_key"] = e0_keep["method_key"].replace({"static-koopman-aligner": "static-koopman-aligner-r48"})

    refresh_pairwise = pd.read_csv(refresh_root / "pairwise_scores.csv")
    refresh_pairwise = refresh_pairwise.loc[refresh_pairwise["method"] == REFRESH_CONTROL_METHOD].copy()

    e2_pairwise = pd.read_csv(e2_root / "pairwise_scores.csv")
    e2_pairwise = e2_pairwise.loc[e2_pairwise["method"] == E2_CONTROL_METHOD].copy()

    panel = pd.concat([e0_keep, refresh_pairwise, e2_pairwise, e2a_pairwise], ignore_index=True)
    return panel.sort_values(["method", "source", "target"]).reset_index(drop=True)


def _build_rbid_panel(
    e0_root: Path,
    e2_root: Path,
    refresh_root: Path,
    method_row: Dict[str, object],
) -> pd.DataFrame:
    e0_rbid = pd.read_csv(e0_root / "phenomenon" / "r48" / "rbid_method_comparison.csv")
    e0_keep = e0_rbid.loc[e0_rbid["method"].isin(["RA", "Static Koopman aligner"])].copy()
    e0_keep["method"] = e0_keep["method"].replace({"Static Koopman aligner": "Static Koopman aligner-r48"})
    e0_keep["method_key"] = e0_keep["method_key"].replace({"static-koopman-aligner": "static-koopman-aligner-r48"})

    refresh_rbid = pd.DataFrame([json.loads((refresh_root / "summary.json").read_text(encoding="utf-8"))["e1_refreshed_pairwise"]])
    e2_rbid = pd.read_csv(e2_root / "rbid_method_comparison.csv")
    e2_rbid = e2_rbid.loc[e2_rbid["method"] == E2_CONTROL_METHOD].copy()
    e2_pairwise = pd.read_csv(e2_root / "pairwise_scores.csv")
    e2_pairwise = e2_pairwise.loc[e2_pairwise["method"] == E2_CONTROL_METHOD].copy()
    if not e2_rbid.empty and not e2_pairwise.empty:
        e2_rbid["pairwise_accuracy_mean"] = float(e2_pairwise["accuracy"].mean())
    e2a_rbid = pd.DataFrame([method_row])

    panel = pd.concat([e0_keep, refresh_rbid, e2_rbid, e2a_rbid], ignore_index=True)
    return panel.sort_values("rbid", ascending=True).reset_index(drop=True)


def _render_memo(summary: Dict[str, object], loso_df: pd.DataFrame, rbid_df: pd.DataFrame) -> str:
    return (
        "# E2a Proxy Replacement Memo\n\n"
        f"**Run dir**: `{summary['run_dir']}`  \n"
        f"**Rank score mode**: `{summary['rank_score_mode']}`\n\n"
        "## LOSO subject results\n\n"
        + loso_df.to_csv(index=False)
        + "\n## RBID comparison\n\n"
        + rbid_df.to_csv(index=False)
        + "\n## Key numbers\n\n"
        + f"- Accuracy mean vs E1: `{summary['delta_vs_e1_loso']['accuracy_mean']:.4f}`\n"
        + f"- RBID vs refresh: `{summary['delta_vs_refresh_pairwise']['rbid']:.4f}`\n"
        + f"- Tail-RBID vs refresh: `{summary['delta_vs_refresh_pairwise']['tail_rbid']:.4f}`\n"
        + f"- Ready for E2b: `{summary['gate']['ready_for_e2b']}`\n"
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
    refresh_root = Path(args.refresh_root)
    behavior_prior = _load_behavior_prior(e0_root)
    run_dir = ensure_dir(Path("results/e2a") / args.run_name)

    start = time.perf_counter()
    folds = _load_fold_cache(loader, all_subjects, target_subjects, pre, cov_eps)
    loso_df, _, _ = evaluate_e2a_loso(
        folds=folds,
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg=DEFAULT_ALIGNER_CFG,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        behavior_prior=behavior_prior,
    )
    e2a_pairwise, e2a_method_row, rank_components = evaluate_e2a_pairwise(
        loader=loader,
        subjects=all_subjects,
        pre=pre,
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg=DEFAULT_ALIGNER_CFG,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
        behavior_prior=behavior_prior,
    )

    e1_summary = json.loads((e1_root / "summary.json").read_text(encoding="utf-8"))
    e2_summary = json.loads((e2_root / "summary.json").read_text(encoding="utf-8"))
    e2_pairwise = pd.read_csv(e2_root / "pairwise_scores.csv")
    e2_pairwise = e2_pairwise.loc[e2_pairwise["method"] == E2_CONTROL_METHOD].copy()
    if not e2_pairwise.empty:
        e2_summary["rbid_aware"]["pairwise_accuracy_mean"] = float(e2_pairwise["accuracy"].mean())
    refresh_summary = json.loads((refresh_root / "summary.json").read_text(encoding="utf-8"))
    pairwise_panel = _build_pairwise_panel(e0_root, e2_root, refresh_root, e2a_pairwise)
    rbid_panel = _build_rbid_panel(e0_root, e2_root, refresh_root, e2a_method_row)
    summary = build_summary(
        loso_df=loso_df,
        method_row=e2a_method_row,
        e1_summary=e1_summary,
        e2_summary=e2_summary,
        refresh_summary=refresh_summary,
        run_dir=run_dir,
    )
    summary["elapsed_sec"] = float(time.perf_counter() - start)
    memo_text = _render_memo(summary, loso_df, rbid_panel)
    write_outputs(
        output_dir=run_dir,
        loso_df=loso_df,
        pairwise_df=pairwise_panel,
        rbid_df=rbid_panel,
        rank_components_df=rank_components,
        summary=summary,
        memo_text=memo_text,
    )
    logger.info("Wrote E2a proxy replacement outputs to %s", run_dir)


if __name__ == "__main__":
    main()
