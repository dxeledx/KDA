import numpy as np
import pandas as pd
import pytest

from src.evaluation.kcar_analysis import (
    compare_window_scores,
    compute_kcar,
    fit_koopman_operator,
    label_window_alignment_risk,
)
from src.evaluation.kcar_policy import (
    apply_ra_first_policy,
    build_budgeted_policy_benchmark,
    compare_policies_against_baseline,
    make_near_causal_scores,
    select_threshold_for_training_windows,
    summarize_budget_curves,
)


def test_compute_kcar_is_bounded_and_has_expected_direction():
    source_better = compute_kcar(
        source_residuals=np.array([0.2, 0.3, 0.4]),
        target_residuals=np.array([0.5, 0.6, 0.7]),
    )
    target_better = compute_kcar(
        source_residuals=np.array([0.6, 0.7, 0.8]),
        target_residuals=np.array([0.2, 0.3, 0.4]),
    )

    assert -1.0 <= source_better <= 1.0
    assert -1.0 <= target_better <= 1.0
    assert source_better < 0.0
    assert target_better > 0.0


def test_fit_koopman_operator_predicts_state_transitions():
    states = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ],
        dtype=np.float64,
    )

    operator = fit_koopman_operator(states, ridge_alpha=1.0e-6)
    lifted = operator.transform(states[:-1])
    predicted = (operator.matrix @ lifted.T).T
    target = operator.transform(states[1:])

    assert operator.matrix.shape[0] == predicted.shape[1]
    assert np.mean(np.linalg.norm(predicted - target, axis=1)) < 0.2


def test_label_window_alignment_risk_covers_all_three_states():
    beneficial = label_window_alignment_risk(
        acc_ra=0.40,
        acc_w0=0.30,
        acc_w05=0.50,
        window_size=32,
    )
    safe = label_window_alignment_risk(
        acc_ra=0.60,
        acc_w0=0.50,
        acc_w05=0.55,
        window_size=32,
    )
    neutral = label_window_alignment_risk(
        acc_ra=0.50,
        acc_w0=0.49,
        acc_w05=0.50,
        window_size=32,
    )

    assert beneficial["label"] == "deviation-beneficial"
    assert safe["label"] == "ra-safe"
    assert neutral["label"] == "neutral"


def test_compare_window_scores_counts_kcar_subject_wins():
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1, 2, 2, 2, 2],
            "window_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "label": [
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "ra-safe",
            ],
            "delta_dev_vs_ra": [0.15, -0.08, 0.12, -0.10, 0.14, -0.09, 0.11, -0.07],
            "rho_kcar": [0.9, 0.1, 0.8, 0.2, 0.7, 0.2, 0.75, 0.1],
            "d_src": [0.55, 0.40, 0.58, 0.41, 0.52, 0.45, 0.50, 0.44],
            "d_tgt": [0.62, 0.60, 0.61, 0.59, 0.58, 0.55, 0.57, 0.56],
            "sigma_recent": [0.20, 0.19, 0.22, 0.20, 0.18, 0.19, 0.21, 0.20],
            "entropy": [0.51, 0.49, 0.53, 0.47, 0.50, 0.48, 0.49, 0.47],
            "conf_max": [0.60, 0.63, 0.61, 0.64, 0.62, 0.65, 0.61, 0.66],
        }
    )

    comparison_df, summary = compare_window_scores(
        frame,
        score_columns=["rho_kcar", "d_src", "d_tgt", "sigma_recent", "entropy", "conf_max"],
    )

    assert set(comparison_df["score"]) == {
        "rho_kcar",
        "d_src",
        "d_tgt",
        "sigma_recent",
        "entropy",
        "conf_max",
    }
    assert summary["subject_wins_vs_heuristics"]["sigma_recent"] == 2
    assert summary["eligible_window_count"] == 8


def test_ra_first_policy_threshold_selection_and_application():
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 1, 2, 2, 2],
            "window_id": [0, 1, 2, 0, 1, 2],
            "label": [
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "deviation-beneficial",
                "ra-safe",
                "ra-safe",
            ],
            "rho_kcar": [0.8, 0.1, 0.9, 0.7, 0.2, 0.3],
            "acc_ra": [0.40, 0.55, 0.42, 0.41, 0.57, 0.58],
            "acc_w05": [0.55, 0.48, 0.56, 0.54, 0.50, 0.51],
        }
    )

    tau = select_threshold_for_training_windows(frame, score_column="rho_kcar")
    assert 0.3 < tau < 0.8

    policy = apply_ra_first_policy(frame, score_column="rho_kcar", threshold=tau)
    assert set(policy["selected_action"]) <= {"stay_with_ra", "use_partial_alignment"}
    assert policy["selected_accuracy"].mean() >= frame["acc_ra"].mean()


def test_make_near_causal_scores_only_uses_past_windows():
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1],
            "window_id": [0, 1, 2, 3],
            "rho_kcar_retro": [0.2, 0.7, 0.4, 0.9],
        }
    )

    causal = make_near_causal_scores(frame, score_column="rho_kcar_retro")

    assert list(causal["rho_kcar_causal"]) == [0.2, 0.2, 0.7, 0.4]


def test_build_budgeted_policy_benchmark_matches_budget_and_policies():
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1, 2, 2, 2, 2],
            "window_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "rho_kcar_retro": [0.9, 0.1, 0.8, 0.2, 0.7, 0.2, 0.75, 0.1],
            "rho_kcar_causal": [0.9, 0.9, 0.1, 0.8, 0.7, 0.7, 0.2, 0.75],
            "d_tgt_retro": [0.6, 0.2, 0.55, 0.25, 0.58, 0.22, 0.56, 0.2],
            "d_tgt_causal": [0.6, 0.6, 0.2, 0.55, 0.58, 0.58, 0.22, 0.56],
            "sigma_recent_retro": [0.3, 0.1, 0.28, 0.08, 0.27, 0.09, 0.26, 0.07],
            "sigma_recent_causal": [0.3, 0.3, 0.1, 0.28, 0.27, 0.27, 0.09, 0.26],
            "acc_ra": [0.40, 0.55, 0.42, 0.58, 0.41, 0.57, 0.43, 0.59],
            "acc_w05": [0.55, 0.48, 0.56, 0.49, 0.54, 0.50, 0.53, 0.51],
            "label": [
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "ra-safe",
                "deviation-beneficial",
                "ra-safe",
            ],
        }
    )

    policy_windows = build_budgeted_policy_benchmark(
        frame,
        policies={"kcar": "rho_kcar", "d_tgt": "d_tgt", "sigma_recent": "sigma_recent"},
        settings=("retrospective", "near_causal"),
        coverages=(0.5,),
    )

    assert set(policy_windows["policy"]) == {"kcar", "d_tgt", "sigma_recent", "ra"}
    assert set(policy_windows["setting"]) == {"retrospective", "near_causal"}
    subset = policy_windows[(policy_windows["policy"] == "kcar") & (policy_windows["setting"] == "retrospective")]
    expected_partial = round(len(frame) * 0.5)
    assert subset["use_partial_alignment"].sum() == expected_partial
    assert subset["selected_action"].isin(["stay_with_ra", "use_partial_alignment"]).all()


def test_policy_summary_and_comparison_capture_strong_controls():
    policy_windows = pd.DataFrame(
        {
            "subject": [1, 1, 2, 2, 1, 1, 2, 2],
            "window_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "policy": ["kcar", "kcar", "kcar", "kcar", "d_tgt", "d_tgt", "d_tgt", "d_tgt"],
            "setting": ["retrospective"] * 8,
            "coverage": [0.5] * 8,
            "selected_accuracy": [0.55, 0.58, 0.54, 0.57, 0.50, 0.54, 0.49, 0.55],
            "acc_ra": [0.40, 0.55, 0.41, 0.57, 0.40, 0.55, 0.41, 0.57],
            "selected_delta_vs_ra": [0.15, 0.03, 0.13, 0.0, 0.10, -0.01, 0.08, -0.02],
            "selected_action": ["use_partial_alignment", "stay_with_ra"] * 4,
            "use_partial_alignment": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    ra_windows = pd.DataFrame(
        {
            "subject": [1, 1, 2, 2],
            "window_id": [0, 1, 0, 1],
            "policy": ["ra"] * 4,
            "setting": ["retrospective"] * 4,
            "coverage": [0.5] * 4,
            "selected_accuracy": [0.40, 0.55, 0.41, 0.57],
            "acc_ra": [0.40, 0.55, 0.41, 0.57],
            "selected_delta_vs_ra": [0.0, 0.0, 0.0, 0.0],
            "selected_action": ["stay_with_ra"] * 4,
            "use_partial_alignment": [0] * 4,
        }
    )
    frame = pd.concat([policy_windows, ra_windows], ignore_index=True)

    subject_summary, summary = summarize_budget_curves(frame)
    comparison = compare_policies_against_baseline(frame, baseline_policy="ra")

    assert {"kcar", "d_tgt", "ra"} <= set(subject_summary["policy"])
    assert summary["kcar"]["retrospective"]["worst_subject_delta"] >= 0.0
    assert set(comparison["policy"]) == {"kcar", "d_tgt"}
