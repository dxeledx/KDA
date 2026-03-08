import numpy as np


def test_non_overlapping_window_slices_cover_sequence():
    from src.evaluation.ksda_v3 import build_window_slices

    windows = build_window_slices(70, 16)
    assert windows[0] == (0, 16)
    assert windows[1] == (16, 32)
    assert windows[-1] == (64, 70)


def test_oracle_usage_ratio_detects_collapse():
    from src.evaluation.ksda_v3 import oracle_usage_stats

    collapsed = oracle_usage_stats(np.array([1, 1, 1, 1, 1]))
    mixed = oracle_usage_stats(np.array([0, 1, 1, 2, 0]))

    assert collapsed["most_common_ratio"] == 1.0
    assert mixed["unique_count"] == 3
    assert mixed["most_common_ratio"] < 0.8


def test_window_oracle_selects_best_action_per_window():
    from src.evaluation.ksda_v3 import compute_window_oracle_actions

    scores = np.array(
        [
            [0.4, 0.5, 0.1],
            [0.2, 0.1, 0.7],
            [0.8, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    chosen, acc = compute_window_oracle_actions(scores)
    assert np.allclose(chosen, np.array([1, 2, 0]))
    assert np.isclose(acc, np.mean([0.5, 0.7, 0.8]))


def test_windowed_action_weights_apply_previous_window_choice():
    from src.evaluation.ksda_v3 import expand_window_actions_to_trials

    actions = np.array([1.0, 0.5, 0.75], dtype=np.float64)
    expanded = expand_window_actions_to_trials(actions, total_trials=70, window_size=32)
    assert np.allclose(expanded[:32], 1.0)
    assert np.allclose(expanded[32:64], 0.5)
    assert np.allclose(expanded[64:], 0.75)


def test_linear_multiclass_selector_recovers_easy_labels():
    from src.evaluation.ksda_v3 import fit_linear_multiclass_selector, predict_linear_multiclass_selector

    X = np.array(
        [
            [2.0, 0.0],
            [1.5, 0.1],
            [-2.0, 0.0],
            [-1.5, -0.1],
            [0.0, 2.0],
            [0.1, 1.5],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    model = fit_linear_multiclass_selector(X, y, num_classes=3, reg_lambda=1.0e-6)
    pred, _ = predict_linear_multiclass_selector(model, X)
    assert np.allclose(pred, y)


def test_linear_scalar_proxy_tracks_monotonic_target():
    from src.evaluation.ksda_v3 import fit_linear_scalar_proxy, predict_linear_scalar_proxy

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    model = fit_linear_scalar_proxy(X, y, reg_lambda=1.0e-6)
    pred = predict_linear_scalar_proxy(model, X)
    assert np.corrcoef(pred, y)[0, 1] > 0.99
