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
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import cka
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
    ]

    correlations = []
    for method, method_name in methods:
        logger.info("Computing phenomenon metrics for %s...", method_name)
        baseline_dir = Path(f"results/baselines/{method}")
        baseline_dir.mkdir(parents=True, exist_ok=True)

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

        rep_values = rep_sim[~np.eye(n, dtype=bool)]
        beh_values = acc_mat[~np.eye(n, dtype=bool)]
        r, p = pearsonr(rep_values, beh_values)
        correlations.append(float(r))

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

        logger.info("%s correlation: r=%.3f p=%.3g", method_name, r, p)

    plot_correlation_comparison(
        [name for _m, name in methods],
        correlations,
        save_path="results/figures/correlation_comparison.pdf",
    )

    logger.info("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
