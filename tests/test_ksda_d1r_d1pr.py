import numpy as np


def test_class_distance_ratio_improves_for_well_separated_features():
    from experiments.ksda_exp_d1r import class_distance_ratio

    rng = np.random.RandomState(0)
    X_bad = rng.randn(40, 6)
    y = np.repeat(np.arange(4), 10)

    centers = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    X_good = np.vstack([centers[label] + 0.1 * rng.randn(10, 6) for label in range(4)])

    assert class_distance_ratio(X_good, y) > class_distance_ratio(X_bad, y)


def test_window_kcar_policy_uses_previous_window_risk_only():
    from experiments.ksda_exp_d1plus_r_signal_benchmark import compute_window_kcar_weights

    rho_t = np.array([0.2] * 32 + [-0.3] * 32 + [0.4] * 16, dtype=np.float64)
    weights = compute_window_kcar_weights(rho_t, window_size=32, low_weight=0.5, high_weight=1.0)

    assert np.allclose(weights[:32], 1.0)
    assert np.allclose(weights[32:64], 0.5)
    assert np.allclose(weights[64:], 1.0)
