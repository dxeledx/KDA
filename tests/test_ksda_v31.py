import numpy as np


def test_build_trial_safe_actions_has_13_actions():
    from src.evaluation.ksda_v31 import build_trial_safe_actions

    rng = np.random.RandomState(0)
    psi_source = rng.randn(40, 12)
    psi_target = rng.randn(20, 12)
    y_source = np.tile(np.arange(4), 10)

    actions = build_trial_safe_actions(psi_source, psi_target, y_source)
    assert len(actions) == 13
    assert "A0_identity" in actions
    assert "P1_mean_shift_a033" in actions
    assert "P4_supervised_subspace_a100" in actions


def test_history_based_action_is_causal_wrt_future_trials():
    from src.evaluation.ksda_v31 import build_trial_safe_actions

    rng = np.random.RandomState(1)
    psi_source = rng.randn(30, 8)
    psi_target_train = rng.randn(10, 8)
    psi_test = rng.randn(12, 8)
    y_source = np.repeat(np.arange(3), 10)

    action = build_trial_safe_actions(psi_source, psi_target_train, y_source)["P1_mean_shift_a100"]
    out_a, _ = action.transform_target_sequence(psi_test)

    psi_test_modified = psi_test.copy()
    psi_test_modified[8:] += 10.0
    out_b, _ = action.transform_target_sequence(psi_test_modified)

    assert np.allclose(out_a[:8], out_b[:8])


def test_causal_teacher_is_previous_window_oracle():
    from src.evaluation.ksda_v31 import build_causal_teacher_actions

    oracle = np.array([3, 2, 1, 0], dtype=np.int64)
    teacher = build_causal_teacher_actions(oracle, default_action=0)
    assert np.allclose(teacher, np.array([0, 3, 2, 1]))


def test_trialize_window_actions_expands_piecewise_constant_labels():
    from src.evaluation.ksda_v31 import trialize_window_actions

    actions = np.array([0, 2, 1], dtype=np.int64)
    expanded = trialize_window_actions(actions, total_trials=10, window_size=4)
    assert np.allclose(expanded[:4], 0)
    assert np.allclose(expanded[4:8], 2)
    assert np.allclose(expanded[8:], 1)


def test_rank_scan_summary_counts_high_overlap_actions():
    from src.evaluation.ksda_v31 import summarize_rank_scan_metrics

    summary = summarize_rank_scan_metrics(
        best_single_action="P2_diag_scaling_a100",
        best_single_accuracy=0.45,
        overlap_vs_identity={
            "P1_mean_shift_a033": 0.96,
            "P1_mean_shift_a067": 0.91,
            "P4_supervised_subspace_a100": 0.99,
        },
    )
    assert summary["num_high_overlap_actions"] == 2
    assert summary["max_overlap_vs_identity"] == 0.99


def test_resolve_representation_config_applies_overrides():
    from src.evaluation.ksda_v31 import resolve_representation_config

    base = {"pca_rank": 16, "lifting": "quadratic"}
    resolved = resolve_representation_config(base, pca_rank=48)
    assert resolved == {"pca_rank": 48, "lifting": "quadratic"}
