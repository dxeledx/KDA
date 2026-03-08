import numpy as np

from src.alignment.behavior_feedback import BehaviorGuidedFeedback


def test_feedback_cold_start_no_adjust():
    fb = BehaviorGuidedFeedback(window_size=10)
    history = [{"conf": 0.8, "entropy": 0.5}] * 5
    w_new = fb.adjust_weight(0.5, history)
    assert np.isclose(w_new, 0.5)


def test_feedback_output_range():
    fb = BehaviorGuidedFeedback(window_size=10, entropy_high_factor=0.8, entropy_low_factor=0.3)
    history = []
    for i in range(10):
        history.append({"conf": 0.9 - 0.01 * i, "entropy": 1.2})

    w_new = fb.adjust_weight(0.5, history)
    assert 0.0 <= w_new <= 1.0


def test_feedback_entropy_priority_resolves_conflicting_rules():
    fb = BehaviorGuidedFeedback(
        window_size=10,
        entropy_high_factor=0.8,
        entropy_low_factor=0.3,
        conf_trend_threshold=-0.01,
        beta=0.05,
        conf_trend_alpha=0.15,
        conflict_mode="entropy_priority",
        momentum=0.0,
    )
    history = []
    for i in range(10):
        history.append({"conf": 0.95 - 0.03 * i, "entropy": 0.1})

    w_new = fb.adjust_weight(0.5, history)
    assert w_new < 0.5
