#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.euclidean import apply_alignment, compute_alignment_matrix
from src.alignment.koopman_alignment import KoopmanFeatureProjector, build_supervised_aligner
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import cka
from src.evaluation.rbid import compute_rbid_from_pairwise
from src.evaluation.visualization import (
    plot_correlation_comparison,
    plot_covariance_heatmaps,
    plot_scatter,
)
from src.features.covariance import compute_covariances, mean_covariance
from src.features.csp import CSP
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("phenomenon_verification")

DEFAULT_CLASSICAL_ROOT = Path("results/baselines")
DEFAULT_OUTPUT_ROOT = Path("results/figures")
DEFAULT_KOOPMAN_LIFTING = "quadratic"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-root", default=str(DEFAULT_CLASSICAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--koopman-pca-rank", type=int, default=16)
    return parser


def _load_transfer_matrix(baseline_dir: Path, key: str) -> np.ndarray:
    path = baseline_dir / f"transfer_{key}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline output: {path}")
    return np.load(path)


def _load_metric_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_metric_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0}
        for col in ["accuracy", "kappa", "f1_macro"]
        if col in df.columns
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_domain_mean(
    loader: BCIDataLoader,
    subjects: List[int],
    pre: Preprocessor,
    method: str,
    cov_eps: float,
) -> Dict[int, np.ndarray]:
    from pyriemann.utils.mean import mean_riemann

    domain_mean = {}
    for sid in subjects:
        X_train, y_train = loader.load_subject(sid, split="train")
        X_train = pre.fit(X_train, y_train).transform(X_train)
        covs = compute_covariances(X_train, eps=cov_eps)
        if method == "ea":
            domain_mean[sid] = mean_covariance(covs)
        elif method == "ra":
            domain_mean[sid] = mean_riemann(covs)
        else:
            raise ValueError(method)
    return domain_mean


def _compute_koopman_pairwise_transfer(
    loader: BCIDataLoader,
    subjects: List[int],
    pre: Preprocessor,
    lda_kwargs: Dict,
    cov_eps: float,
    mode: str,
    pca_rank: int,
    lifting: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.evaluation.metrics import compute_metrics
    from src.models.classifiers import LDA

    n = len(subjects)
    acc = np.zeros((n, n), dtype=np.float64)
    kappa = np.zeros((n, n), dtype=np.float64)
    f1 = np.zeros((n, n), dtype=np.float64)

    for i, src in enumerate(subjects):
        X_src_train, y_src_train = loader.load_subject(src, split="train")
        X_src_test, _ = loader.load_subject(src, split="test")
        X_src_train = pre.fit(X_src_train, y_src_train).transform(X_src_train)
        X_src_test = pre.transform(X_src_test)
        cov_src_train = compute_covariances(X_src_train, eps=cov_eps)
        cov_src_test = compute_covariances(X_src_test, eps=cov_eps)
        projector = KoopmanFeatureProjector(pca_rank=pca_rank, lifting=lifting).fit(cov_src_train)
        psi_src_train = projector.transform(cov_src_train)
        psi_src_test = projector.transform(cov_src_test)

        aligner = None
        if mode == "static-koopman-aligner":
            aligner = build_supervised_aligner(
                "A1", k=32, reg_lambda=1.0e-4, normalize_output=True
            ).fit(psi_src_train, y_src_train)
            psi_src_train = aligner.transform(psi_src_train)
            psi_src_test = aligner.transform(psi_src_test)

        lda = LDA(**lda_kwargs).fit(psi_src_train, y_src_train)
        for j, tgt in enumerate(subjects):
            X_tgt_test, y_tgt_test = loader.load_subject(tgt, split="test")
            X_tgt_test = pre.transform(X_tgt_test)
            cov_tgt_test = compute_covariances(X_tgt_test, eps=cov_eps)
            psi_tgt_test = projector.transform(cov_tgt_test)
            if aligner is not None:
                psi_tgt_test = aligner.transform(psi_tgt_test)
            y_pred = lda.predict(psi_tgt_test)
            metrics = compute_metrics(y_tgt_test, y_pred)
            acc[i, j] = metrics["accuracy"]
            kappa[i, j] = metrics["kappa"]
            f1[i, j] = metrics["f1_macro"]
    return acc, kappa, f1


def _compute_koopman_loso(
    loader: BCIDataLoader,
    subjects: List[int],
    pre: Preprocessor,
    lda_kwargs: Dict,
    cov_eps: float,
    mode: str,
    pca_rank: int,
    lifting: str,
) -> pd.DataFrame:
    from src.evaluation.metrics import compute_metrics
    from src.models.classifiers import LDA

    rows = []
    for target in subjects:
        X_source_blocks, y_source_blocks = [], []
        for subject in subjects:
            if subject == target:
                continue
            X_train_subject, y_train_subject = loader.load_subject(subject, split="train")
            X_source_blocks.append(X_train_subject)
            y_source_blocks.append(y_train_subject)

        X_source = np.concatenate(X_source_blocks, axis=0)
        y_source = np.concatenate(y_source_blocks, axis=0)
        X_target_train, _ = loader.load_subject(target, split="train")
        X_target_test, y_target_test = loader.load_subject(target, split="test")

        X_source = pre.fit(X_source, y_source).transform(X_source)
        X_target_train = pre.transform(X_target_train)
        X_target_test = pre.transform(X_target_test)

        cov_source = compute_covariances(X_source, eps=cov_eps)
        cov_target_train = compute_covariances(X_target_train, eps=cov_eps)
        cov_target_test = compute_covariances(X_target_test, eps=cov_eps)

        projector = KoopmanFeatureProjector(pca_rank=pca_rank, lifting=lifting).fit(cov_source)
        psi_source = projector.transform(cov_source)
        psi_target_train = projector.transform(cov_target_train)
        psi_target_test = projector.transform(cov_target_test)

        if mode == "static-koopman-aligner":
            aligner = build_supervised_aligner(
                "A1", k=32, reg_lambda=1.0e-4, normalize_output=True
            ).fit(psi_source, y_source)
            psi_source = aligner.transform(psi_source)
            _ = aligner.transform(psi_target_train)
            psi_target_test = aligner.transform(psi_target_test)

        lda = LDA(**lda_kwargs).fit(psi_source, y_source)
        y_pred = lda.predict(psi_target_test)
        metrics = compute_metrics(y_target_test, y_pred)
        rows.append({"target_subject": target, **metrics})

    return pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True)


def _phenomenon_methods(pca_rank: int) -> List[Tuple[str, str]]:
    _ = pca_rank
    return [
        ("noalign", "No Alignment"),
        ("ea", "EA"),
        ("ra", "RA"),
        ("koopman-noalign", "Koopman-noalign"),
        ("static-koopman-aligner", "Static Koopman aligner"),
    ]


def _loso_summary_row(
    method_name: str,
    method_key: str,
    loso_df: pd.DataFrame,
    pca_rank: int | None,
    lifting: str | None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method_name,
        "method_key": method_key,
        "accuracy_mean": float(loso_df["accuracy"].mean()),
        "accuracy_std": float(loso_df["accuracy"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
        "kappa_mean": float(loso_df["kappa"].mean()),
        "kappa_std": float(loso_df["kappa"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
        "f1_macro_mean": float(loso_df["f1_macro"].mean()),
        "f1_macro_std": float(loso_df["f1_macro"].std(ddof=1)) if len(loso_df) > 1 else 0.0,
        "koopman_pca_rank": pca_rank,
        "lifting": lifting,
    }
    return row


def _methods_root(output_root: Path) -> Path:
    if output_root == DEFAULT_OUTPUT_ROOT:
        return DEFAULT_CLASSICAL_ROOT
    return ensure_dir(output_root / "methods")


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    classical_root = Path(args.classical_root)
    output_root = ensure_dir(args.output_root)
    methods_root = _methods_root(output_root)
    koopman_pca_rank = int(args.koopman_pca_rank)
    koopman_lifting = DEFAULT_KOOPMAN_LIFTING

    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))

    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    subjects = list(map(int, data_cfg["dataset"]["subjects"]))

    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))

    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))
    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]

    # Covariance heatmaps (A01, A05, A09)
    cov_subjects = [1, 5, 9]
    cov_means = []
    cov_labels = []
    for sid in cov_subjects:
        X_train, y_train = loader.load_subject(sid, split="train")
        X_train = pre.fit(X_train, y_train).transform(X_train)
        covs = compute_covariances(X_train, eps=cov_eps)
        cov_means.append(mean_covariance(covs))
        cov_labels.append(f"A{sid:02d}")
    plot_covariance_heatmaps(
        cov_means,
        cov_labels,
        save_path=output_root / "covariance_heatmaps.pdf",
    )

    methods = _phenomenon_methods(koopman_pca_rank)
    correlations: List[float] = []
    rbid_rows = []
    loso_rows = []
    all_pair_rows = []

    for method, method_name in methods:
        logger.info("Computing phenomenon metrics for %s...", method_name)
        if method in ("noalign", "ea", "ra"):
            baseline_dir = classical_root / method
        else:
            baseline_dir = ensure_dir(methods_root / method)

        if method in ("koopman-noalign", "static-koopman-aligner"):
            acc_mat, kappa_mat, f1_mat = _compute_koopman_pairwise_transfer(
                loader,
                subjects,
                pre,
                lda_kwargs,
                cov_eps,
                mode=method,
                pca_rank=koopman_pca_rank,
                lifting=koopman_lifting,
            )
            np.save(baseline_dir / "transfer_accuracy.npy", acc_mat)
            np.save(baseline_dir / "transfer_kappa.npy", kappa_mat)
            np.save(baseline_dir / "transfer_f1_macro.npy", f1_mat)
            labels = [f"A{s:02d}" for s in subjects]
            pd.DataFrame(acc_mat, index=labels, columns=labels).to_csv(
                baseline_dir / "transfer_accuracy.csv"
            )
            pd.DataFrame(kappa_mat, index=labels, columns=labels).to_csv(
                baseline_dir / "transfer_kappa.csv"
            )
            pd.DataFrame(f1_mat, index=labels, columns=labels).to_csv(
                baseline_dir / "transfer_f1_macro.csv"
            )
            loso_df = _compute_koopman_loso(
                loader,
                subjects,
                pre,
                lda_kwargs,
                cov_eps,
                mode=method,
                pca_rank=koopman_pca_rank,
                lifting=koopman_lifting,
            )
            loso_df.to_csv(baseline_dir / "loso.csv", index=False)
            _save_metric_summary(loso_df, baseline_dir / "summary.json")
            loso_rows.append(
                _loso_summary_row(method_name, method, loso_df, koopman_pca_rank, koopman_lifting)
            )
        else:
            acc_mat = _load_transfer_matrix(baseline_dir, "accuracy")
            kappa_mat = _load_transfer_matrix(baseline_dir, "kappa")
            f1_mat = _load_transfer_matrix(baseline_dir, "f1_macro")
            summary = _load_metric_summary(baseline_dir / "summary.json")
            loso_rows.append(
                {
                    "method": method_name,
                    "method_key": method,
                    "accuracy_mean": float(summary["accuracy"]["mean"]),
                    "accuracy_std": float(summary["accuracy"]["std"]),
                    "kappa_mean": float(summary["kappa"]["mean"]),
                    "kappa_std": float(summary["kappa"]["std"]),
                    "f1_macro_mean": float(summary["f1_macro"]["mean"]),
                    "f1_macro_std": float(summary["f1_macro"]["std"]),
                    "koopman_pca_rank": None,
                    "lifting": None,
                }
            )

        domain_mean = None
        if method in ("ea", "ra"):
            domain_mean = _compute_domain_mean(
                loader, subjects, pre, method=method, cov_eps=cov_eps
            )

        n = len(subjects)
        rep_sim = np.eye(n, dtype=np.float64)
        pair_rows = []

        for i, src in enumerate(subjects):
            X_src_train, y_src_train = loader.load_subject(src, split="train")
            X_src_test, _ = loader.load_subject(src, split="test")
            X_src_train = pre.fit(X_src_train, y_src_train).transform(X_src_train)
            X_src_test = pre.transform(X_src_test)

            if method in ("koopman-noalign", "static-koopman-aligner"):
                cov_src_train = compute_covariances(X_src_train, eps=cov_eps)
                cov_src_test = compute_covariances(X_src_test, eps=cov_eps)
                projector = KoopmanFeatureProjector(
                    pca_rank=koopman_pca_rank, lifting=koopman_lifting
                ).fit(cov_src_train)
                psi_src_train = projector.transform(cov_src_train)
                F_src = projector.transform(cov_src_test)
                aligner = None
                if method == "static-koopman-aligner":
                    aligner = build_supervised_aligner(
                        "A1", k=32, reg_lambda=1.0e-4, normalize_output=True
                    ).fit(psi_src_train, y_src_train)
                    F_src = aligner.transform(F_src)
            else:
                csp = CSP(**csp_kwargs).fit(X_src_train, y_src_train)
                F_src = csp.transform(X_src_test)

            for j, tgt in enumerate(subjects):
                if src == tgt:
                    continue

                X_tgt_test, y_tgt_test = loader.load_subject(tgt, split="test")
                X_tgt_test = pre.transform(X_tgt_test)

                if method in ("ea", "ra"):
                    assert domain_mean is not None
                    A = compute_alignment_matrix(
                        domain_mean[src], domain_mean[tgt], eps=cov_eps
                    )
                    X_eval = apply_alignment(X_tgt_test, A)
                else:
                    X_eval = X_tgt_test

                if method in ("koopman-noalign", "static-koopman-aligner"):
                    cov_tgt_test = compute_covariances(X_eval, eps=cov_eps)
                    F_tgt = projector.transform(cov_tgt_test)
                    if aligner is not None:
                        F_tgt = aligner.transform(F_tgt)
                else:
                    F_tgt = csp.transform(X_eval)
                rep = cka(F_src, F_tgt)
                rep_sim[i, j] = rep

                pair_rows.append(
                    {
                        "source": src,
                        "target": tgt,
                        "accuracy": float(acc_mat[i, j]),
                        "kappa": float(kappa_mat[i, j]),
                        "f1_macro": float(f1_mat[i, j]),
                        "cka": float(rep),
                    }
                )

        np.save(baseline_dir / "rep_sim.npy", rep_sim)
        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(baseline_dir / "pair_metrics.csv", index=False)
        pair_df.assign(method=method).to_csv(baseline_dir / "pair_metrics_tagged.csv", index=False)
        all_pair_rows.append(pair_df.assign(method=method_name, method_key=method))

        rep_values = rep_sim[~np.eye(n, dtype=bool)]
        beh_values = acc_mat[~np.eye(n, dtype=bool)]
        r, p = pearsonr(rep_values, beh_values)
        correlations.append(float(r))
        rbid = compute_rbid_from_pairwise(rep_sim, acc_mat)
        rbid_rows.append(
            {
                "method": method_name,
                "method_key": method,
                "pearson_r": float(r),
                "rbid": float(rbid["rbid"]),
                "rbid_pos": float(rbid["rbid_pos"]),
                "rbid_neg": float(rbid["rbid_neg"]),
                "tail_rbid": float(rbid["tail_rbid"]),
            }
        )

        plot_scatter(
            rep_values,
            beh_values,
            save_path=output_root / f"rep_acc_scatter_{method}.pdf",
            title=f"Representation-Behavior Inconsistency ({method_name})",
            r=float(r),
            p_value=float(p),
        )

        case_a = pair_df[(pair_df["cka"] > 0.7) & (pair_df["accuracy"] < 0.5)]
        case_b = pair_df[(pair_df["cka"] < 0.4) & (pair_df["accuracy"] > 0.65)]

        lines = []
        lines.append("Case A: high CKA but low transfer accuracy (cka>0.7 & acc<0.5)")
        for row in case_a.sort_values(["cka", "accuracy"], ascending=[False, True]).itertuples():
            lines.append(
                f"  A{row.source:02d} -> A{row.target:02d}: CKA={row.cka:.3f}, acc={row.accuracy:.3f}"
            )
        lines.append("")
        lines.append("Case B: low CKA but high transfer accuracy (cka<0.4 & acc>0.65)")
        for row in case_b.sort_values(["cka", "accuracy"], ascending=[True, False]).itertuples():
            lines.append(
                f"  A{row.source:02d} -> A{row.target:02d}: CKA={row.cka:.3f}, acc={row.accuracy:.3f}"
            )

        (baseline_dir / "inconsistent_cases.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        (baseline_dir / "correlation.json").write_text(
            json.dumps({"r": float(r), "p_value": float(p)}, indent=2), encoding="utf-8"
        )
        rbid["pair_df"].to_csv(baseline_dir / "rbid_pairs.csv", index=False)
        logger.info("%s correlation: r=%.3f p=%.3g", method_name, r, p)

    plot_correlation_comparison(
        [name for _m, name in methods],
        correlations,
        save_path=output_root / "correlation_comparison.pdf",
    )

    rbid_df = pd.DataFrame(rbid_rows).sort_values("rbid", ascending=False)
    pairwise_df = pd.concat(all_pair_rows, ignore_index=True)
    loso_summary_df = pd.DataFrame(loso_rows)
    pairwise_df.to_csv(output_root / "pairwise_scores.csv", index=False)
    rbid_df.to_csv(output_root / "rbid_method_comparison.csv", index=False)
    rbid_df[["method", "rbid", "rbid_pos", "rbid_neg", "tail_rbid"]].to_csv(
        output_root / "rbid_summary.csv", index=False
    )
    rbid_df[["method", "rbid_pos", "rbid_neg"]].to_csv(
        output_root / "rbid_direction_breakdown.csv", index=False
    )
    loso_summary_df.to_csv(output_root / "loso_method_summary.csv", index=False)
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "classical_root": str(classical_root),
                "output_root": str(output_root),
                "koopman_pca_rank": koopman_pca_rank,
                "lifting": koopman_lifting,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(8.0, 4.6))
    plt.scatter(rbid_df["rbid"], rbid_df["pearson_r"], s=80, alpha=0.8)
    for row in rbid_df.itertuples(index=False):
        plt.annotate(
            row.method,
            (row.rbid, row.pearson_r),
            fontsize=9,
            xytext=(4, 2),
            textcoords="offset points",
        )
    plt.xlabel("RBID")
    plt.ylabel("Pearson r")
    plt.title("RBID vs Pearson")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "rbid_scatter.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(8.0, 4.6))
    plt.bar(rbid_df["method"], rbid_df["tail_rbid"])
    plt.ylabel("Tail-RBID")
    plt.title("Tail-RBID across methods")
    plt.xticks(rotation=20, ha="right")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_root / "rbid_tail_bar.pdf", dpi=300)
    plt.close()

    logger.info("Done. Phenomenon outputs saved to %s", output_root)


if __name__ == "__main__":
    main()
