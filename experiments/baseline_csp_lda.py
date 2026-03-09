#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.evaluation.protocols import evaluate_loso, evaluate_pairwise_transfer, evaluate_within_subject
from src.evaluation.visualization import plot_transfer_matrix
from src.utils.config import ensure_dir, load_yaml, seed_everything
from src.utils.logger import get_logger


logger = get_logger("baseline_csp_lda")


def _save_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in ["accuracy", "kappa", "f1_macro"]
        if col in df.columns
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/baselines/noalign")
    parser.add_argument("--fig-dir", default="results/figures")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    exp_cfg = load_yaml("configs/experiment_config.yaml")
    seed_everything(int(exp_cfg["experiment"]["seed"]))

    data_cfg = load_yaml("configs/data_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")

    loader = BCIDataLoader.from_config(data_cfg)
    subjects = list(map(int, data_cfg["dataset"]["subjects"]))

    norm_cfg = data_cfg["preprocessing"]["normalize"]
    pre = Preprocessor(normalize=bool(norm_cfg["enabled"]), eps=float(norm_cfg["eps"]))

    csp_kwargs = model_cfg["csp"]
    lda_kwargs = model_cfg["lda"]
    cov_eps = float(model_cfg.get("alignment", {}).get("eps", 1.0e-6))

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)

    logger.info("Running within-subject evaluation...")
    within_df = evaluate_within_subject(loader, subjects, pre, csp_kwargs, lda_kwargs)
    within_df.to_csv(out_dir / "within_subject.csv", index=False)

    logger.info("Running LOSO evaluation (no alignment)...")
    loso_df = evaluate_loso(
        loader, subjects, pre, csp_kwargs, lda_kwargs, method="noalign", cov_eps=cov_eps
    )
    loso_df.to_csv(out_dir / "loso.csv", index=False)
    _save_summary(loso_df, out_dir / "summary.json")

    logger.info("Running pairwise transfer matrix (no alignment)...")
    matrices = evaluate_pairwise_transfer(
        loader, subjects, pre, csp_kwargs, lda_kwargs, method="noalign", cov_eps=cov_eps
    )
    for key, mat in matrices.items():
        np.save(out_dir / f"transfer_{key}.npy", mat)
        labels = [f"A{s:02d}" for s in subjects]
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(
            out_dir / f"transfer_{key}.csv"
        )

    plot_transfer_matrix(
        matrices["accuracy"],
        fig_dir / "transfer_matrix_noalign.pdf",
        title="Cross-Subject Transfer Performance (No Alignment)",
    )

    logger.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
