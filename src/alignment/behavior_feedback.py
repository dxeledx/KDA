from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from scipy.stats import linregress


class BehaviorGuidedFeedback:
    """Behavior-guided adjustment of the alignment weight.

    Uses prediction entropy / confidence trend as unsupervised signals to
    increase/decrease alignment strength.
    """

    def __init__(
        self,
        window_size: int = 10,
        n_classes: int = 4,
        entropy_high_factor: float = 0.8,
        entropy_low_factor: float = 0.3,
        conf_trend_threshold: float = -0.05,
        alpha: float = 0.1,
        beta: float = 0.05,
        conf_trend_alpha: float = 0.15,
        momentum: float = 0.7,
        delta_w_max: float = 0.2,
        update_every: int = 1,
        conflict_mode: str = "sum",
    ):
        self.window_size = int(window_size)
        self.n_classes = int(n_classes)
        self.H_threshold_high = float(entropy_high_factor) * math.log(float(self.n_classes))
        self.H_threshold_low = float(entropy_low_factor) * math.log(float(self.n_classes))
        self.conf_trend_threshold = float(conf_trend_threshold)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.conf_trend_alpha = float(conf_trend_alpha)
        self.momentum = float(momentum)
        self.delta_w_max = float(delta_w_max)
        self.update_every = int(update_every)
        self.conflict_mode = str(conflict_mode)

        self.velocity = 0.0

    def reset(self) -> None:
        self.velocity = 0.0

    def adjust_weight(self, w_pred: float, history: List[Dict]) -> float:
        if len(history) < self.window_size:
            return float(np.clip(w_pred, 0.0, 1.0))

        if self.update_every > 1 and (len(history) % self.update_every != 0):
            return float(np.clip(w_pred, 0.0, 1.0))

        metrics = self._compute_metrics(history[-self.window_size :])
        entropy_delta = 0.0
        confidence_delta = 0.0

        # Rule 1: entropy feedback
        if metrics["H_avg"] > self.H_threshold_high:
            entropy_delta += self.alpha
        elif metrics["H_avg"] < self.H_threshold_low:
            entropy_delta -= self.beta

        # Rule 2: confidence trend feedback
        if metrics["conf_trend"] < self.conf_trend_threshold:
            confidence_delta += self.conf_trend_alpha

        delta_w = self._resolve_rule_deltas(entropy_delta, confidence_delta)

        delta_w = float(np.clip(delta_w, -self.delta_w_max, self.delta_w_max))
        self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * delta_w

        w_new = float(w_pred) + float(self.velocity)
        return float(np.clip(w_new, 0.0, 1.0))

    def _compute_metrics(self, window: List[Dict]) -> Dict[str, float]:
        entropies = [float(h.get("entropy", 0.0)) for h in window]
        H_avg = float(np.mean(entropies)) if entropies else 0.0

        confidences = [float(h.get("conf", 0.0)) for h in window]
        conf_trend = 0.0
        if len(confidences) >= 5:
            slope, _intercept, _r, _p, _stderr = linregress(
                np.arange(len(confidences), dtype=np.float64), np.asarray(confidences)
            )
            conf_trend = float(slope)

        return {"H_avg": H_avg, "conf_trend": conf_trend}

    def _resolve_rule_deltas(self, entropy_delta: float, confidence_delta: float) -> float:
        if abs(entropy_delta) <= 1.0e-12:
            return float(confidence_delta)
        if abs(confidence_delta) <= 1.0e-12:
            return float(entropy_delta)

        same_direction = np.sign(entropy_delta) == np.sign(confidence_delta)
        if same_direction or self.conflict_mode == "sum":
            return float(entropy_delta + confidence_delta)
        if self.conflict_mode == "entropy_priority":
            return float(entropy_delta)
        if self.conflict_mode == "confidence_priority":
            return float(confidence_delta)
        if self.conflict_mode == "average":
            return float(0.5 * (entropy_delta + confidence_delta))
        raise ValueError(f"Unsupported conflict_mode: {self.conflict_mode}")
