import numpy as np


def test_compute_kcar_window_is_near_zero_when_operators_match():
    from experiments.ksda_exp_d1_plus_kcar_gate import compute_kcar_window
    from src.evaluation.kcar_analysis import fit_koopman_operator

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
    rho = compute_kcar_window(states, operator, operator)
    assert abs(rho) < 1.0e-6


def test_kcar_gate_is_monotonic_in_risk():
    from experiments.ksda_exp_d1_plus_kcar_gate import kcar_gate

    low_risk = kcar_gate(-0.5)
    mid_risk = kcar_gate(0.0)
    high_risk = kcar_gate(0.5)

    assert low_risk > mid_risk > high_risk
    assert 0.0 <= high_risk <= 1.0
    assert 0.0 <= low_risk <= 1.0
