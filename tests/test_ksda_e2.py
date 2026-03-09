import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _balanced_labels(n: int, n_classes: int = 4) -> np.ndarray:
    labels = np.arange(n, dtype=np.int64) % n_classes
    return labels


def _make_covariances(rng: np.random.RandomState, n_trials: int, n_channels: int = 6, n_samples: int = 48) -> np.ndarray:
    from src.features.covariance import compute_covariances

    X = rng.randn(n_trials, n_channels, n_samples)
    return compute_covariances(X, eps=1.0e-6)


def test_conservative_aligner_requires_behavior_prior_when_rank_enabled():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    rng = np.random.RandomState(0)
    source = rng.randn(24, 10)
    target = rng.randn(12, 10)
    y_source = _balanced_labels(len(source))

    with pytest.raises(ValueError, match="behavior_prior_scores"):
        KoopmanConservativeResidualAligner(
            residual_rank=4,
            basis_k=6,
            lambda_dyn=0.0,
            lambda_rank=1.0,
            max_iter=3,
        ).fit(
            source,
            target,
            y_source=y_source,
            source_subject_ids=[1, 2],
            source_block_lengths=[12, 12],
        )


def test_conservative_aligner_rank_loss_prefers_correct_order():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    rng = np.random.RandomState(1)
    source = rng.randn(24, 10)
    target = rng.randn(12, 10)
    y_source = _balanced_labels(len(source))

    aligner = KoopmanConservativeResidualAligner(
        residual_rank=4,
        basis_k=6,
        lambda_dyn=0.0,
        lambda_rank=1.0,
        max_iter=0,
    ).fit(
        source,
        target,
        y_source=y_source,
        source_subject_ids=[1, 2],
        source_block_lengths=[12, 12],
        behavior_prior_scores={1: 0.8, 2: 0.2},
    )

    good = aligner.compute_rank_loss(np.array([0.8, 0.2]), np.array([0.7, 0.3]))
    bad = aligner.compute_rank_loss(np.array([0.8, 0.2]), np.array([0.3, 0.7]))
    assert good < bad


def test_e2_parser_accepts_run_and_roots():
    from experiments import ksda_exp_e2_rbid_aware_aligner

    parser = ksda_exp_e2_rbid_aware_aligner.build_argparser()
    args = parser.parse_args(["--run-name", "demo", "--e0-root", "results/e0/demo", "--e1-root", "results/e1/demo"])
    assert args.run_name == "demo"
    assert args.e0_root == "results/e0/demo"
    assert args.e1_root == "results/e1/demo"


def test_e2_single_fold_smoke_and_output_files(tmp_path: Path):
    from experiments.ksda_exp_e2_rbid_aware_aligner import (
        DEFAULT_ALIGNER_CFG,
        DEFAULT_REP_CFG,
        build_summary,
        evaluate_rbid_aware_loso,
        write_outputs,
    )

    rng = np.random.RandomState(0)
    fold = {
        "target_subject": 3,
        "cov_source": _make_covariances(rng, 24),
        "y_source": _balanced_labels(24),
        "cov_target_train": _make_covariances(rng, 12),
        "cov_target_test": _make_covariances(rng, 12),
        "y_target_test": _balanced_labels(12),
        "source_block_lengths": [12, 12],
        "source_subject_ids": [1, 2],
    }
    behavior_prior = {3: {1: 0.7, 2: 0.4}}

    loso_df, pairwise_df, method_row = evaluate_rbid_aware_loso(
        folds=[fold],
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg={**DEFAULT_ALIGNER_CFG, "lambda_dyn": 0.0, "max_iter": 3, "lambda_rank": 1.0},
        lda_kwargs={"solver": "lsqr", "shrinkage": "auto"},
        cov_eps=1.0e-6,
        behavior_prior=behavior_prior,
    )

    assert len(loso_df) == 1
    assert len(pairwise_df) == 1
    assert method_row["method"] == "RBID-aware Conservative Koopman aligner"

    baseline_main = pd.DataFrame(
        [
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
            }
        ]
    )
    e1_summary = {
        "conservative": {
            "method": "Conservative Koopman aligner-r48",
            "accuracy_mean": 0.4248,
            "rbid": 0.3056,
            "tail_rbid": 0.5503,
        }
    }
    rbid_df = pd.DataFrame(
        [
            {
                "method": "RBID-aware Conservative Koopman aligner-r48",
                "method_key": "rbid-aware-conservative-koopman-aligner-r48",
                "rbid": float(method_row["rbid"]),
                "rbid_pos": float(method_row["rbid_pos"]),
                "rbid_neg": float(method_row["rbid_neg"]),
                "tail_rbid": float(method_row["tail_rbid"]),
                "pearson_r": float(method_row["pearson_r"]),
            }
        ]
    )

    summary = build_summary(
        loso_df=loso_df,
        method_row=method_row,
        baseline_main=baseline_main,
        e1_summary=e1_summary,
        control_method="Static Koopman aligner-r48",
        run_dir=tmp_path,
    )
    write_outputs(
        output_dir=tmp_path,
        loso_df=loso_df,
        pairwise_df=pairwise_df,
        rbid_df=rbid_df,
        summary=summary,
        memo_text="demo e2",
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "loso_subject_results.csv").exists()
    assert (tmp_path / "pairwise_scores.csv").exists()
    assert (tmp_path / "rbid_method_comparison.csv").exists()
    assert (tmp_path / "memo.md").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert "delta_vs_e1" in saved
    assert saved["control_e1"]["method"] == "Conservative Koopman aligner-r48"
