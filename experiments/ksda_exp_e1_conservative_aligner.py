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


logger = get_logger("ksda_exp_e1_conservative_aligner")

DEFAULT_REP_CFG = {"pca_rank": 48, "lifting": "quadratic"}
DEFAULT_ALIGNER_CFG = {
    "basis_method": "A1",
    "basis_k": 32,
    "basis_reg_lambda": 1.0e-4,
    "residual_rank": 8,
    "lambda_cls": 1.0,
    "lambda_dyn": 1.0,
    "lambda_reg": 1.0e-3,
    "ridge_alpha": 1.0e-3,
    "max_iter": 100,
}
CONTROL_METHOD = "Static Koopman aligner-r48"
E0_METHODS = ["EA", "RA", "Koopman-noalign-r48", CONTROL_METHOD]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="2026-03-09-e1-r1")
    parser.add_argument("--e0-root", default="results/e0/2026-03-09-e0-r1")
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


def _make_aligner(aligner_cfg: Dict[str, object]) -> KoopmanConservativeResidualAligner:
    return KoopmanConservativeResidualAligner(
        residual_rank=int(aligner_cfg["residual_rank"]),
        basis_k=int(aligner_cfg["basis_k"]),
        basis_reg_lambda=float(aligner_cfg["basis_reg_lambda"]),
        lambda_cls=float(aligner_cfg["lambda_cls"]),
        lambda_dyn=float(aligner_cfg["lambda_dyn"]),
        lambda_reg=float(aligner_cfg["lambda_reg"]),
        ridge_alpha=float(aligner_cfg["ridge_alpha"]),
        max_iter=int(aligner_cfg["max_iter"]),
    )


def evaluate_conservative_loso(
    folds: Sequence[Dict[str, np.ndarray | int | list[int]]],
    rep_cfg: Dict[str, object],
    aligner_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
    cov_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rows = []
    pair_rows = []
    for fold in folds:
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
            source_block_lengths=fold["source_block_lengths"],
        )
        psi_source_aligned = aligner.transform_source(psi_source)
        psi_target_test_aligned = aligner.transform_target(psi_target_test)

        lda = LDA(**lda_kwargs).fit(psi_source_aligned, fold["y_source"])
        y_pred = lda.predict(psi_target_test_aligned)
        metrics = compute_metrics(fold["y_target_test"], y_pred)
        rows.append(
            {
                "target_subject": int(fold["target_subject"]),
                **metrics,
                "optimizer_success": bool(aligner.optimization_info_["success"]),
                "optimizer_nit": int(aligner.optimization_info_["nit"]),
                "loss_final": float(aligner.optimization_info_.get("fun", 0.0)),
            }
        )
        pair_rows.append(
            {
                "source": 0,
                "target": int(fold["target_subject"]),
                "accuracy": float(metrics["accuracy"]),
                "kappa": float(metrics["kappa"]),
                "f1_macro": float(metrics["f1_macro"]),
                "cka": float(cka(psi_source_aligned[: len(psi_target_test_aligned)], psi_target_test_aligned[: len(psi_source_aligned[: len(psi_target_test_aligned)])])),
                "method": "Conservative Koopman aligner-r48",
                "method_key": "conservative-koopman-aligner-r48",
            }
        )

    loso_df = pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)
    pairwise_df = pd.DataFrame(pair_rows).sort_values("target").reset_index(drop=True)
    method_row = {
        "method": "Conservative Koopman aligner",
        "method_key": "conservative-koopman-aligner-r48",
        "accuracy_mean": float(loso_df["accuracy"].mean()),
        "accuracy_std": float(loso_df["accuracy"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
        "kappa_mean": float(loso_df["kappa"].mean()),
        "kappa_std": float(loso_df["kappa"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
        "rbid": 0.0,
        "rbid_pos": 0.0,
        "rbid_neg": 0.0,
        "tail_rbid": 0.0,
        "pearson_r": 0.0,
    }
    return loso_df, pairwise_df, method_row


def evaluate_conservative_pairwise(
    loader,
    subjects: Sequence[int],
    pre,
    rep_cfg: Dict[str, object],
    aligner_cfg: Dict[str, object],
    lda_kwargs: Dict[str, object],
    cov_eps: float,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    n = len(subjects)
    acc = np.eye(n, dtype=np.float64)
    kappa = np.eye(n, dtype=np.float64)
    f1 = np.eye(n, dtype=np.float64)
    rep = np.eye(n, dtype=np.float64)
    rows = []

    for i, src in enumerate(subjects):
        X_source_train, y_source_train = loader.load_subject(int(src), split="train")
        X_source_test, _ = loader.load_subject(int(src), split="test")
        X_source_train = pre.fit(X_source_train, y_source_train).transform(X_source_train)
        X_source_test = pre.transform(X_source_test)

        cov_source_train = compute_covariances(X_source_train, eps=cov_eps)
        cov_source_test = compute_covariances(X_source_test, eps=cov_eps)
        projector = KoopmanFeatureProjector(
            pca_rank=int(rep_cfg["pca_rank"]),
            lifting=str(rep_cfg["lifting"]),
        ).fit(cov_source_train)
        psi_source_train = projector.transform(cov_source_train)
        psi_source_test = projector.transform(cov_source_test)

        for j, tgt in enumerate(subjects):
            if src == tgt:
                continue
            X_target_train, _ = loader.load_subject(int(tgt), split="train")
            X_target_test, y_target_test = loader.load_subject(int(tgt), split="test")
            X_target_train = pre.transform(X_target_train)
            X_target_test = pre.transform(X_target_test)

            cov_target_train = compute_covariances(X_target_train, eps=cov_eps)
            cov_target_test = compute_covariances(X_target_test, eps=cov_eps)
            psi_target_train = projector.transform(cov_target_train)
            psi_target_test = projector.transform(cov_target_test)

            aligner = _make_aligner(aligner_cfg).fit(
                psi_source_train,
                psi_target_train,
                y_source=y_source_train,
                source_block_lengths=[len(psi_source_train)],
            )
            psi_source_train_aligned = aligner.transform_source(psi_source_train)
            psi_source_test_aligned = aligner.transform_source(psi_source_test)
            psi_target_test_aligned = aligner.transform_target(psi_target_test)

            lda = LDA(**lda_kwargs).fit(psi_source_train_aligned, y_source_train)
            y_pred = lda.predict(psi_target_test_aligned)
            metrics = compute_metrics(y_target_test, y_pred)

            acc[i, j] = metrics["accuracy"]
            kappa[i, j] = metrics["kappa"]
            f1[i, j] = metrics["f1_macro"]
            rep[i, j] = cka(psi_source_test_aligned, psi_target_test_aligned)
            rows.append(
                {
                    "source": int(src),
                    "target": int(tgt),
                    "accuracy": float(metrics["accuracy"]),
                    "kappa": float(metrics["kappa"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "cka": float(rep[i, j]),
                    "method": "Conservative Koopman aligner-r48",
                    "method_key": "conservative-koopman-aligner-r48",
                }
            )

    rbid = compute_rbid_from_pairwise(rep, acc)
    row = {
        "method": "Conservative Koopman aligner-r48",
        "method_key": "conservative-koopman-aligner-r48",
        "rbid": float(rbid["rbid"]),
        "rbid_pos": float(rbid["rbid_pos"]),
        "rbid_neg": float(rbid["rbid_neg"]),
        "tail_rbid": float(rbid["tail_rbid"]),
    }
    rep_values = rep[~np.eye(n, dtype=bool)]
    beh_values = acc[~np.eye(n, dtype=bool)]
    if np.allclose(rep_values, rep_values[0]) or np.allclose(beh_values, beh_values[0]):
        row["pearson_r"] = 0.0
    else:
        row["pearson_r"] = float(np.corrcoef(rep_values, beh_values)[0, 1])
    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True), row


def build_summary(
    *,
    loso_df: pd.DataFrame,
    method_row: Dict[str, object],
    baseline_main: pd.DataFrame,
    control_method: str,
    run_dir: Path,
) -> Dict[str, object]:
    control = baseline_main.loc[baseline_main["method"] == control_method].iloc[0].to_dict()
    accuracy_mean = float(loso_df["accuracy"].mean())
    rbid = float(method_row["rbid"])
    summary = {
        "run_dir": str(run_dir),
        "representation": dict(DEFAULT_REP_CFG),
        "control": {"method": control_method, **control},
        "conservative": {
            "method": "Conservative Koopman aligner-r48",
            "accuracy_mean": accuracy_mean,
            "accuracy_std": float(loso_df["accuracy"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
            "kappa_mean": float(loso_df["kappa"].mean()),
            "kappa_std": float(loso_df["kappa"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
            "rbid": rbid,
            "rbid_pos": float(method_row["rbid_pos"]),
            "rbid_neg": float(method_row["rbid_neg"]),
            "tail_rbid": float(method_row["tail_rbid"]),
            "pearson_r": float(method_row["pearson_r"]),
        },
        "delta_vs_control": {
            "accuracy_mean": accuracy_mean - float(control["accuracy_mean"]),
            "rbid": rbid - float(control["rbid"]),
            "tail_rbid": float(method_row["tail_rbid"]) - float(control["tail_rbid"]),
        },
        "gate": {
            "accuracy_pass": bool(accuracy_mean >= 0.3835),
            "rbid_pass": bool(rbid < 0.3135),
            "ready_for_e2": bool(accuracy_mean >= 0.3835 and rbid < 0.3135),
        },
    }
    return summary


def write_outputs(
    *,
    output_dir: Path,
    loso_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    rbid_df: pd.DataFrame,
    summary: Dict[str, object],
    memo_text: str,
) -> None:
    ensure_dir(output_dir)
    loso_df.to_csv(output_dir / "loso_subject_results.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_scores.csv", index=False)
    rbid_df.to_csv(output_dir / "rbid_method_comparison.csv", index=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "memo.md").write_text(memo_text, encoding="utf-8")


def _build_pairwise_panel(e0_root: Path, conservative_pairwise: pd.DataFrame) -> pd.DataFrame:
    e0_pairwise = pd.read_csv(e0_root / "phenomenon" / "r48" / "pairwise_scores.csv")
    keep_methods = {
        "EA": "EA",
        "RA": "RA",
        "Koopman-noalign": "Koopman-noalign-r48",
        "Static Koopman aligner": "Static Koopman aligner-r48",
    }
    baseline = e0_pairwise.loc[e0_pairwise["method"].isin(keep_methods)].copy()
    baseline["method"] = baseline["method"].map(keep_methods)
    baseline["method_key"] = baseline["method_key"].replace(
        {
            "koopman-noalign": "koopman-noalign-r48",
            "static-koopman-aligner": "static-koopman-aligner-r48",
        }
    )
    panel = pd.concat([baseline, conservative_pairwise], ignore_index=True)
    return panel.sort_values(["method", "source", "target"]).reset_index(drop=True)


def _build_rbid_panel(e0_root: Path, conservative_method_row: Dict[str, object]) -> pd.DataFrame:
    e0_rbid = pd.read_csv(e0_root / "phenomenon" / "r48" / "rbid_method_comparison.csv")
    keep_methods = {
        "EA": "EA",
        "RA": "RA",
        "Koopman-noalign": "Koopman-noalign-r48",
        "Static Koopman aligner": "Static Koopman aligner-r48",
    }
    baseline = e0_rbid.loc[e0_rbid["method"].isin(keep_methods)].copy()
    baseline["method"] = baseline["method"].map(keep_methods)
    baseline["method_key"] = baseline["method_key"].replace(
        {
            "koopman-noalign": "koopman-noalign-r48",
            "static-koopman-aligner": "static-koopman-aligner-r48",
        }
    )
    conservative = pd.DataFrame([conservative_method_row])
    panel = pd.concat([baseline, conservative], ignore_index=True)
    return panel.sort_values("rbid", ascending=True).reset_index(drop=True)


def _render_memo(summary: Dict[str, object], loso_df: pd.DataFrame, rbid_df: pd.DataFrame) -> str:
    control = summary["control"]
    conservative = summary["conservative"]
    gate = summary["gate"]
    return (
        "# E1 Conservative Koopman Aligner Memo\n\n"
        f"**Control**: `{control['method']}`  \n"
        f"**Run dir**: `{summary['run_dir']}`  \n"
        f"**Representation**: `pca_rank={summary['representation']['pca_rank']}`, `lifting={summary['representation']['lifting']}`\n\n"
        "## LOSO subject results\n\n"
        + loso_df.to_csv(index=False)
        + "\n## RBID comparison\n\n"
        + rbid_df.to_csv(index=False)
        + "\n## Key numbers\n\n"
        + f"- Conservative accuracy mean: `{conservative['accuracy_mean']:.4f}`\n"
        + f"- Control accuracy mean: `{control['accuracy_mean']:.4f}`\n"
        + f"- Conservative RBID: `{conservative['rbid']:.4f}`\n"
        + f"- Control RBID: `{control['rbid']:.4f}`\n"
        + f"- Gate ready for E2: `{gate['ready_for_e2']}`\n"
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

    run_dir = ensure_dir(Path("results/e1") / args.run_name)
    start = time.perf_counter()
    folds = _load_fold_cache(loader, all_subjects, target_subjects, pre, cov_eps)
    loso_df, _, _ = evaluate_conservative_loso(
        folds=folds,
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg=DEFAULT_ALIGNER_CFG,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
    )
    conservative_pairwise, conservative_method_row = evaluate_conservative_pairwise(
        loader=loader,
        subjects=all_subjects,
        pre=pre,
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg=DEFAULT_ALIGNER_CFG,
        lda_kwargs=lda_kwargs,
        cov_eps=cov_eps,
    )
    e0_root = Path(args.e0_root)
    baseline_main = pd.read_csv(e0_root / "summary" / "e0_main_table.csv")
    pairwise_panel = _build_pairwise_panel(e0_root, conservative_pairwise)
    rbid_panel = _build_rbid_panel(e0_root, conservative_method_row)
    summary = build_summary(
        loso_df=loso_df,
        method_row=conservative_method_row,
        baseline_main=baseline_main,
        control_method=CONTROL_METHOD,
        run_dir=run_dir,
    )
    summary["elapsed_sec"] = float(time.perf_counter() - start)
    memo_text = _render_memo(summary, loso_df, rbid_panel)
    write_outputs(
        output_dir=run_dir,
        loso_df=loso_df,
        pairwise_df=pairwise_panel,
        rbid_df=rbid_panel,
        summary=summary,
        memo_text=memo_text,
    )
    logger.info("E1 results saved to %s", run_dir)


if __name__ == "__main__":
    main()
