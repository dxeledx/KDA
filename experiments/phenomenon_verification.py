#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_transfer_matrix(baseline_dir: Path, key: str) -> np.ndarray:
    path = baseline_dir / f"transfer_{key}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline output: {path}")
    return np.load(path)


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
    pca_rank: int = 16,
    lifting: str = "quadratic",
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


def main() -> None:
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

    ensure_dir("results/figures")

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
        save_path="results/figures/covariance_heatmaps.pdf",
    )

    methods: List[Tuple[str, str]] = [
        ("noalign", "No Alignment"),
        ("ea", "EA"),
        ("ra", "RA"),
        ("koopman-noalign", "Koopman-noalign"),
        ("static-koopman-aligner", "Static Koopman aligner"),
    ]

    correlations = []
    rbid_rows = []
    all_pair_rows = []
    for method, method_name in methods:
        logger.info("Computing phenomenon metrics for %s...", method_name)
        baseline_dir = Path(f"results/baselines/{method}")
        baseline_dir.mkdir(parents=True, exist_ok=True)

        if method in ("koopman-noalign", "static-koopman-aligner"):
            lda_kwargs = model_cfg["lda"]
            acc_mat, kappa_mat, f1_mat = _compute_koopman_pairwise_transfer(
                loader,
                subjects,
                pre,
                lda_kwargs,
                cov_eps,
                mode=method,
            )
            np.save(baseline_dir / "transfer_accuracy.npy", acc_mat)
            np.save(baseline_dir / "transfer_kappa.npy", kappa_mat)
            np.save(baseline_dir / "transfer_f1_macro.npy", f1_mat)
        else:
            acc_mat = _load_transfer_matrix(baseline_dir, "accuracy")
            kappa_mat = _load_transfer_matrix(baseline_dir, "kappa")
            f1_mat = _load_transfer_matrix(baseline_dir, "f1_macro")

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
            X_src_test, y_src_test = loader.load_subject(src, split="test")
            X_src_train = pre.fit(X_src_train, y_src_train).transform(X_src_train)
            X_src_test = pre.transform(X_src_test)

            if method in ("koopman-noalign", "static-koopman-aligner"):
                cov_src_train = compute_covariances(X_src_train, eps=cov_eps)
                cov_src_test = compute_covariances(X_src_test, eps=cov_eps)
                projector = KoopmanFeatureProjector(pca_rank=16, lifting="quadratic").fit(cov_src_train)
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
            save_path=f"results/figures/rep_acc_scatter_{method}.pdf",
            title=f"Representation-Behavior Inconsistency ({method_name})",
            r=float(r),
            p_value=float(p),
        )

        # Inconsistent cases
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
        save_path="results/figures/correlation_comparison.pdf",
    )

    rbid_df = pd.DataFrame(rbid_rows).sort_values("rbid", ascending=False)
    pd.concat(all_pair_rows, ignore_index=True).to_csv("results/figures/pairwise_scores.csv", index=False)
    rbid_df.to_csv("results/figures/rbid_method_comparison.csv", index=False)
    rbid_df[["method", "rbid", "rbid_pos", "rbid_neg", "tail_rbid"]].to_csv(
        "results/figures/rbid_summary.csv", index=False
    )
    rbid_df[["method", "rbid_pos", "rbid_neg"]].to_csv(
        "results/figures/rbid_direction_breakdown.csv", index=False
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.0, 4.6))
    plt.scatter(rbid_df["rbid"], rbid_df["pearson_r"], s=80, alpha=0.8)
    for row in rbid_df.itertuples(index=False):
        plt.annotate(row.method, (row.rbid, row.pearson_r), fontsize=9, xytext=(4, 2), textcoords="offset points")
    plt.xlabel("RBID")
    plt.ylabel("Pearson r")
    plt.title("RBID vs Pearson")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/rbid_scatter.pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(8.0, 4.6))
    plt.bar(rbid_df["method"], rbid_df["tail_rbid"])
    plt.ylabel("Tail-RBID")
    plt.title("Tail-RBID across methods")
    plt.xticks(rotation=20, ha="right")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/figures/rbid_tail_bar.pdf", dpi=300)
    plt.close()

    logger.info("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
