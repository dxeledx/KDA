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

from src.alignment.koopman_alignment import fit_alignment, transform
from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.metrics import compute_metrics
from src.evaluation.protocols import evaluate_loso
from src.features.covariance import compute_covariances
from src.models.classifiers import LDA
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("ksda_exp_d1")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_targets(all_subjects: List[int], targets: str | None) -> List[int]:
    if not targets:
        return list(all_subjects)
    wanted = {int(token.strip()) for token in targets.split(",") if token.strip()}
    return [subject for subject in all_subjects if subject in wanted]


def _summarize_against_ra(method_df: pd.DataFrame, ra_df: pd.DataFrame, method: str, elapsed_sec: float) -> Dict:
    delta = method_df["accuracy"].to_numpy(dtype=np.float64) - ra_df["accuracy"].to_numpy(dtype=np.float64)
    return {
        "method": method,
        "accuracy_mean": float(method_df["accuracy"].mean()),
        "accuracy_std": float(method_df["accuracy"].std(ddof=1)) if len(method_df) > 1 else 0.0,
        "delta_vs_ra": float(delta.mean()),
        "wins": int(np.sum(delta > 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "draws": int(np.sum(np.isclose(delta, 0.0))),
        "elapsed_sec": elapsed_sec,
    }


def _run_koopman_loso(
    loader: BCIDataLoader,
    all_subjects: List[int],
    target_subjects: List[int],
    pre: Preprocessor,
    lda_kwargs: Dict,
    cov_eps: float,
    pca_rank: int,
    align: bool,
    details_dir: Path | None = None,
) -> tuple[pd.DataFrame, float]:
    rows = []
    start = time.perf_counter()
    for target in target_subjects:
        X_source_blocks, y_source_blocks = [], []
        for subject in all_subjects:
            if subject == target:
                continue
            X_train_subject, y_train_subject = loader.load_subject(subject, split="train")
            X_source_blocks.append(X_train_subject)
            y_source_blocks.append(y_train_subject)

        X_source_raw = np.concatenate(X_source_blocks, axis=0)
        y_source = np.concatenate(y_source_blocks, axis=0)
        X_target_train_raw, _ = loader.load_subject(target, split="train")
        X_target_test_raw, y_target_test = loader.load_subject(target, split="test")

        X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
        X_target_train = pre.transform(X_target_train_raw)
        X_target_test = pre.transform(X_target_test_raw)

        cov_source = compute_covariances(X_source, eps=cov_eps)
        cov_target_train = compute_covariances(X_target_train, eps=cov_eps)
        cov_target_test = compute_covariances(X_target_test, eps=cov_eps)

        projector, aligner, psi_source, _ = fit_alignment(
            cov_source, cov_target_train, pca_rank=pca_rank, cov_eps=cov_eps
        )
        psi_test = transform(cov_target_test, projector, aligner if align else None)
        lda = LDA(**lda_kwargs).fit(psi_source, y_source)
        y_pred = lda.predict(psi_test)
        rows.append({"target_subject": target, **compute_metrics(y_target_test, y_pred)})

        if details_dir is not None:
            np.savez(
                details_dir / f"subject_A{target:02d}.npz",
                y_true=np.asarray(y_target_test, dtype=np.int64),
                y_pred=np.asarray(y_pred, dtype=np.int64),
                psi_test=psi_test,
            )
        logger.info("D1 method=%s target=%s acc=%.4f", "ksda-static" if align else "koopman-noalign", target, rows[-1]["accuracy"])

    return pd.DataFrame(rows).sort_values("target_subject").reset_index(drop=True), float(time.perf_counter() - start)


def main() -> None:
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
    target_subjects = _resolve_targets(all_subjects, args.targets)
    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))
    lda_kwargs = model_cfg["lda"]
    csp_kwargs = model_cfg["csp"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    root_dir = ensure_dir(f"results/ksda/exp_d1/{args.run_name}")
    details_dir = ensure_dir(root_dir / "details")
    fig_dir = ensure_dir(root_dir / "figures")

    ra_start = time.perf_counter()
    ra_df = evaluate_loso(loader, all_subjects, pre, csp_kwargs, lda_kwargs, "ra", cov_eps)
    ra_elapsed = float(time.perf_counter() - ra_start)
    noalign_start = time.perf_counter()
    noalign_df = evaluate_loso(loader, all_subjects, pre, csp_kwargs, lda_kwargs, "noalign", cov_eps)
    noalign_elapsed = float(time.perf_counter() - noalign_start)
    koopman_noalign_df, koopman_noalign_elapsed = _run_koopman_loso(
        loader, all_subjects, target_subjects, pre, lda_kwargs, cov_eps, int(args.pca_rank), False
    )
    ksda_df, ksda_elapsed = _run_koopman_loso(
        loader, all_subjects, target_subjects, pre, lda_kwargs, cov_eps, int(args.pca_rank), True, details_dir
    )

    ra_df.to_csv(root_dir / "ra_loso.csv", index=False)
    noalign_df.to_csv(root_dir / "noalign_loso.csv", index=False)
    koopman_noalign_df.to_csv(root_dir / "koopman_noalign_loso.csv", index=False)
    ksda_df.to_csv(root_dir / "loso_results.csv", index=False)

    rows = [
        _summarize_against_ra(ra_df, ra_df, "ra", ra_elapsed),
        _summarize_against_ra(noalign_df, ra_df, "noalign", noalign_elapsed),
        _summarize_against_ra(koopman_noalign_df, ra_df, "koopman-noalign", koopman_noalign_elapsed),
        _summarize_against_ra(ksda_df, ra_df, "ksda-static", ksda_elapsed),
    ]
    comparison_df = pd.DataFrame(rows).sort_values(["accuracy_mean", "wins"], ascending=[False, False])
    comparison_df.to_csv(root_dir / "comparison.csv", index=False)
    _save_json({row["method"]: row for row in rows}, root_dir / "summary.json")
    _save_json(
        {"ra": ra_elapsed, "noalign": noalign_elapsed, "koopman-noalign": koopman_noalign_elapsed, "ksda-static": ksda_elapsed},
        root_dir / "timing.json",
    )

    plt.figure(figsize=(8.5, 4.5))
    order = ["ra", "ksda-static", "koopman-noalign", "noalign"]
    mapping = {"ra": ra_df, "ksda-static": ksda_df, "koopman-noalign": koopman_noalign_df, "noalign": noalign_df}
    data = [mapping[method]["accuracy"].to_numpy(dtype=np.float64) for method in order]
    plt.boxplot(data, labels=order)
    plt.ylabel("Accuracy")
    plt.title("KSDA D.1 vs baselines")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_comparison.pdf", dpi=300)
    plt.close()
    logger.info("Saved D.1 outputs to %s", root_dir)


if __name__ == "__main__":
    main()
