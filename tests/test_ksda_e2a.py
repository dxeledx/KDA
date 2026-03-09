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


def test_conservative_aligner_mean_dyn_rank_scores_prefer_closer_mean():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    source_block_1 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    source_block_2 = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]], dtype=np.float64)
    source = np.vstack([source_block_1, source_block_2])
    target = np.array([[0.2, 0.1], [0.1, 0.2], [0.2, 0.2]], dtype=np.float64)
    y_source = _balanced_labels(len(source), n_classes=2)

    aligner = KoopmanConservativeResidualAligner(
        residual_rank=1,
        basis_k=1,
        lambda_dyn=0.0,
        lambda_rank=1.0,
        rank_score_mode="mean_dyn_neg_l2",
        rank_mean_weight=1.0,
        rank_dyn_weight=0.0,
        max_iter=0,
    ).fit(
        source,
        target,
        y_source=y_source,
        source_subject_ids=[1, 2],
        source_block_lengths=[len(source_block_1), len(source_block_2)],
        behavior_prior_scores={1: 0.8, 2: 0.2},
    )

    components = aligner.compute_rank_score_components(
        source_features=source,
        target_features=target,
        source_block_lengths=[len(source_block_1), len(source_block_2)],
    )
    assert components.shape[0] == 2
    assert float(components.loc[1, "mean_sq_dist"]) < float(components.loc[0, "mean_sq_dist"])
    assert float(components.loc[1, "u_score"]) > float(components.loc[0, "u_score"])


def test_conservative_aligner_mean_dyn_rank_scores_use_dyn_residual():
    from src.alignment.koopman_alignment import KoopmanConservativeResidualAligner

    source_block_1 = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float64)
    source_block_2 = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    source = np.vstack([source_block_1, source_block_2])
    target = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float64)
    y_source = _balanced_labels(len(source), n_classes=2)

    aligner = KoopmanConservativeResidualAligner(
        residual_rank=1,
        basis_k=1,
        lambda_dyn=0.0,
        lambda_rank=1.0,
        rank_score_mode="mean_dyn_neg_l2",
        rank_mean_weight=0.0,
        rank_dyn_weight=1.0,
        max_iter=0,
    ).fit(
        source,
        target,
        y_source=y_source,
        source_subject_ids=[1, 2],
        source_block_lengths=[len(source_block_1), len(source_block_2)],
        behavior_prior_scores={1: 0.8, 2: 0.2},
    )

    components = aligner.compute_rank_score_components(
        source_features=source,
        target_features=target,
        source_block_lengths=[len(source_block_1), len(source_block_2)],
    )
    assert float(components.loc[0, "dyn_resid"]) < float(components.loc[1, "dyn_resid"])
    assert float(components.loc[0, "u_score"]) > float(components.loc[1, "u_score"])


def test_e2a_parser_accepts_all_roots():
    from experiments import ksda_exp_e2a_proxy_replacement

    parser = ksda_exp_e2a_proxy_replacement.build_argparser()
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
            "--refresh-root",
            "results/e2_diag/demo",
        ]
    )
    assert args.run_name == "demo"
    assert args.e0_root == "results/e0/demo"
    assert args.e1_root == "results/e1/demo"
    assert args.e2_root == "results/e2/demo"
    assert args.refresh_root == "results/e2_diag/demo"


def test_e2a_single_fold_smoke_and_outputs(tmp_path: Path):
    from experiments.ksda_exp_e2a_proxy_replacement import (
        DEFAULT_ALIGNER_CFG,
        DEFAULT_REP_CFG,
        build_summary,
        evaluate_e2a_loso,
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

    loso_df, pairwise_df, method_row = evaluate_e2a_loso(
        folds=[fold],
        rep_cfg=DEFAULT_REP_CFG,
        aligner_cfg={**DEFAULT_ALIGNER_CFG, "lambda_dyn": 0.0, "max_iter": 3, "lambda_rank": 1.0},
        lda_kwargs={"solver": "lsqr", "shrinkage": "auto"},
        cov_eps=1.0e-6,
        behavior_prior=behavior_prior,
    )
    assert len(loso_df) == 1
    assert len(pairwise_df) == 1
    assert method_row["method"] == "RBID-aware Conservative Koopman aligner-r48 (mean+dyn proxy)"

    e1_summary = {
        "conservative": {
            "method": "Conservative Koopman aligner-r48",
            "accuracy_mean": 0.4248,
            "kappa_mean": 0.2330,
        }
    }
    e2_summary = {
        "rbid_aware": {
            "method": "RBID-aware Conservative Koopman aligner-r48",
            "rbid": 0.3095,
            "tail_rbid": 0.6234,
            "pearson_r": 0.3696,
            "pairwise_accuracy_mean": 0.3149,
        }
    }
    refresh_summary = {
        "e1_refreshed_pairwise": {
            "method": "Conservative Koopman aligner-r48 (target-global refresh)",
            "rbid": 0.2937,
            "tail_rbid": 0.6190,
            "pearson_r": 0.3579,
            "pairwise_accuracy_mean": 0.3208,
        }
    }
    summary = build_summary(
        loso_df=loso_df,
        method_row={
            **method_row,
            "pairwise_accuracy_mean": 0.31,
            "rbid": 0.30,
            "tail_rbid": 0.60,
            "pearson_r": 0.35,
        },
        e1_summary=e1_summary,
        e2_summary=e2_summary,
        refresh_summary=refresh_summary,
        run_dir=tmp_path,
    )
    write_outputs(
        output_dir=tmp_path,
        loso_df=loso_df,
        pairwise_df=pairwise_df,
        rbid_df=pd.DataFrame([method_row]),
        rank_components_df=pd.DataFrame(
            [{"source": 1, "target": 3, "mean_sq_dist": 0.1, "dyn_resid": 0.2, "u_score": -0.3, "behavior_prior": 0.7, "accuracy": 0.5}]
        ),
        summary=summary,
        memo_text="demo e2a",
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "loso_subject_results.csv").exists()
    assert (tmp_path / "pairwise_scores.csv").exists()
    assert (tmp_path / "rbid_method_comparison.csv").exists()
    assert (tmp_path / "rank_score_components.csv").exists()
    assert (tmp_path / "memo.md").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert "control_pairwise_refresh" in saved
    assert "delta_vs_refresh_pairwise" in saved
    assert saved["rank_score_mode"] == "mean_dyn_neg_l2"
