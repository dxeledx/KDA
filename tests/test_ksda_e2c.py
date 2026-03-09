import json
from pathlib import Path

import numpy as np
import pandas as pd


def _balanced_labels(n: int, n_classes: int = 4) -> np.ndarray:
    return np.arange(n, dtype=np.int64) % n_classes


def _make_covariances(rng: np.random.RandomState, n_trials: int, n_channels: int = 6, n_samples: int = 48) -> np.ndarray:
    from src.features.covariance import compute_covariances

    X = rng.randn(n_trials, n_channels, n_samples)
    return compute_covariances(X, eps=1.0e-6)


def test_e2c_parser_accepts_all_roots():
    from experiments import ksda_exp_e2c_tail_soft_rbid

    parser = ksda_exp_e2c_tail_soft_rbid.build_argparser()
    args = parser.parse_args(
        [
            "--run-name",
            "demo",
            "--e0-root",
            "results/e0/demo",
            "--e1-root",
            "results/e1/demo",
            "--e2-root",
            "results/e2/demo",
            "--e2a-root",
            "results/e2a/demo",
            "--e2b-root",
            "results/e2b/demo",
            "--refresh-root",
            "results/e2_diag/demo",
        ]
    )
    assert args.run_name == "demo"
    assert args.e2b_root == "results/e2b/demo"


def test_e2c_single_fold_smoke_and_outputs(tmp_path: Path):
    from experiments.ksda_exp_e2c_tail_soft_rbid import (
        DEFAULT_ALIGNER_CFG,
        DEFAULT_REP_CFG,
        build_summary,
        evaluate_e2c_loso,
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

    loso_df, pairwise_df, method_row = evaluate_e2c_loso(
        folds=[fold],
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg={**DEFAULT_ALIGNER_CFG, "lambda_dyn": 0.0, "max_iter": 3, "lambda_rank": 1.0},
        lda_kwargs={"solver": "lsqr", "shrinkage": "auto"},
        cov_eps=1.0e-6,
        behavior_prior=behavior_prior,
    )
    assert len(loso_df) == 1
    assert len(pairwise_df) == 1
    assert method_row["method"] == "RBID-aware Conservative Koopman aligner-r48 (Tail-Soft-RBID)"

    e1_summary = {"conservative": {"method": "Conservative Koopman aligner-r48", "accuracy_mean": 0.4248, "kappa_mean": 0.2330}}
    e2_summary = {"rbid_aware": {"method": "RBID-aware Conservative Koopman aligner-r48", "rbid": 0.3095, "tail_rbid": 0.6234, "pearson_r": 0.3696, "pairwise_accuracy_mean": 0.3149}}
    e2a_summary = {"e2a_proxy_replacement": {"method": "RBID-aware Conservative Koopman aligner-r48 (mean+dyn proxy)", "rbid": 0.3095, "tail_rbid": 0.6327, "pearson_r": 0.3696, "pairwise_accuracy_mean": 0.3146}}
    e2b_summary = {"e2b_soft_rbid": {"method": "RBID-aware Conservative Koopman aligner-r48 (Soft-RBID)", "rbid": 0.3095, "tail_rbid": 0.6234, "pearson_r": 0.3712, "pairwise_accuracy_mean": 0.3145}}
    refresh_summary = {"e1_refreshed_pairwise": {"method": "Conservative Koopman aligner-r48 (target-global refresh)", "rbid": 0.2937, "tail_rbid": 0.6190, "pearson_r": 0.3579, "pairwise_accuracy_mean": 0.3208}}
    summary = build_summary(
        loso_df=loso_df,
        method_row={**method_row, "pairwise_accuracy_mean": 0.31, "rbid": 0.30, "tail_rbid": 0.60, "pearson_r": 0.35},
        e1_summary=e1_summary,
        e2_summary=e2_summary,
        e2a_summary=e2a_summary,
        e2b_summary=e2b_summary,
        refresh_summary=refresh_summary,
        run_dir=tmp_path,
    )
    write_outputs(
        output_dir=tmp_path,
        loso_df=loso_df,
        pairwise_df=pairwise_df,
        rbid_df=pd.DataFrame([method_row]),
        summary=summary,
        memo_text="demo e2c",
    )

    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["rank_loss_mode"] == "tail_soft_rbid_huber"
    assert "secondary_control_e2b" in saved
    assert "delta_vs_e2b_pairwise" in saved
