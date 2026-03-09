import json
from pathlib import Path

import numpy as np
import pandas as pd


def _balanced_labels(n: int, n_classes: int = 4) -> np.ndarray:
    labels = np.arange(n, dtype=np.int64) % n_classes
    return labels


def _make_covariances(rng: np.random.RandomState, n_trials: int, n_channels: int = 6, n_samples: int = 48) -> np.ndarray:
    from src.features.covariance import compute_covariances

    X = rng.randn(n_trials, n_channels, n_samples)
    return compute_covariances(X, eps=1.0e-6)


def test_e1_parser_accepts_run_and_e0_roots():
    from experiments import ksda_exp_e1_conservative_aligner

    parser = ksda_exp_e1_conservative_aligner.build_argparser()
    args = parser.parse_args(["--run-name", "demo", "--e0-root", "results/e0/demo"])
    assert args.run_name == "demo"
    assert args.e0_root == "results/e0/demo"


def test_e1_single_fold_smoke_and_output_files(tmp_path: Path):
    from experiments.ksda_exp_e1_conservative_aligner import (
        DEFAULT_ALIGNER_CFG,
        DEFAULT_REP_CFG,
        build_summary,
        evaluate_conservative_loso,
        write_outputs,
    )

    rng = np.random.RandomState(0)
    fold = {
        "target_subject": 1,
        "cov_source": _make_covariances(rng, 24),
        "y_source": _balanced_labels(24),
        "cov_target_train": _make_covariances(rng, 12),
        "cov_target_test": _make_covariances(rng, 12),
        "y_target_test": _balanced_labels(12),
        "source_block_lengths": [12, 12],
    }

    loso_df, pairwise_df, method_row = evaluate_conservative_loso(
        folds=[fold],
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg={**DEFAULT_ALIGNER_CFG, "lambda_dyn": 0.0, "max_iter": 3},
        lda_kwargs={"solver": "lsqr", "shrinkage": "auto"},
        cov_eps=1.0e-6,
    )

    assert len(loso_df) == 1
    assert len(pairwise_df) == 1
    assert method_row["method"] == "Conservative Koopman aligner"

    baseline_main = pd.DataFrame(
        [
            {
                "method": "RA",
                "accuracy_mean": 0.44,
                "accuracy_std": 0.14,
                "kappa_mean": 0.25,
                "kappa_std": 0.18,
                "rbid": 0.28,
                "rbid_pos": 0.14,
                "rbid_neg": 0.14,
                "tail_rbid": 0.62,
                "pearson_r": 0.61,
            },
            {
                "method": "EA",
                "accuracy_mean": 0.43,
                "accuracy_std": 0.14,
                "kappa_mean": 0.24,
                "kappa_std": 0.19,
                "rbid": 0.28,
                "rbid_pos": 0.14,
                "rbid_neg": 0.14,
                "tail_rbid": 0.60,
                "pearson_r": 0.57,
            },
            {
                "method": "Koopman-noalign-r48",
                "accuracy_mean": 0.39,
                "accuracy_std": 0.13,
                "kappa_mean": 0.19,
                "kappa_std": 0.17,
                "rbid": 0.34,
                "rbid_pos": 0.17,
                "rbid_neg": 0.17,
                "tail_rbid": 0.67,
                "pearson_r": 0.20,
            },
            {
                "method": "Static Koopman aligner-r48",
                "accuracy_mean": 0.3835,
                "accuracy_std": 0.1367,
                "kappa_mean": 0.1780,
                "kappa_std": 0.1823,
                "rbid": 0.3135,
                "rbid_pos": 0.1567,
                "rbid_neg": 0.1567,
                "tail_rbid": 0.6357,
                "pearson_r": 0.4841,
            },
        ]
    )
    rbid_df = pd.DataFrame(
        [
            {"method": "RA", "rbid": 0.28, "rbid_pos": 0.14, "rbid_neg": 0.14, "tail_rbid": 0.62, "pearson_r": 0.61},
            {"method": "EA", "rbid": 0.28, "rbid_pos": 0.14, "rbid_neg": 0.14, "tail_rbid": 0.60, "pearson_r": 0.57},
            {"method": "Koopman-noalign-r48", "rbid": 0.34, "rbid_pos": 0.17, "rbid_neg": 0.17, "tail_rbid": 0.67, "pearson_r": 0.20},
            {"method": "Static Koopman aligner-r48", "rbid": 0.3135, "rbid_pos": 0.1567, "rbid_neg": 0.1567, "tail_rbid": 0.6357, "pearson_r": 0.4841},
            {"method": "Conservative Koopman aligner-r48", "rbid": float(method_row["rbid"]), "rbid_pos": float(method_row["rbid_pos"]), "rbid_neg": float(method_row["rbid_neg"]), "tail_rbid": float(method_row["tail_rbid"]), "pearson_r": float(method_row["pearson_r"])},
        ]
    )

    summary = build_summary(
        loso_df=loso_df,
        method_row=method_row,
        baseline_main=baseline_main,
        control_method="Static Koopman aligner-r48",
        run_dir=tmp_path,
    )
    write_outputs(
        output_dir=tmp_path,
        loso_df=loso_df,
        pairwise_df=pairwise_df,
        rbid_df=rbid_df,
        summary=summary,
        memo_text="demo memo",
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "loso_subject_results.csv").exists()
    assert (tmp_path / "pairwise_scores.csv").exists()
    assert (tmp_path / "rbid_method_comparison.csv").exists()
    assert (tmp_path / "memo.md").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["control"]["method"] == "Static Koopman aligner-r48"
    assert "Conservative Koopman aligner-r48" in pd.read_csv(tmp_path / "rbid_method_comparison.csv")["method"].tolist()
