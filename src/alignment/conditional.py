from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class LinearConditionalWeight:
    """Linear + sigmoid mapping from context to alignment weight w_t in [0, 1]."""

    def __init__(
        self,
        weights: Sequence[float],
        bias: float = 0.0,
        temperature: float = 1.0,
        ema_smooth_alpha: float = 0.0,
    ):
        self.weights = np.asarray(list(weights), dtype=np.float64)
        self.bias = float(bias)
        self.temperature = float(temperature)
        self.ema_smooth_alpha = float(ema_smooth_alpha)
        self._w_prev: Optional[float] = None

    def reset(self) -> None:
        self._w_prev = None

    def predict(self, c_t: np.ndarray) -> float:
        c = np.asarray(c_t, dtype=np.float64).ravel()
        if c.shape[0] != self.weights.shape[0]:
            raise ValueError(
                f"Context dim mismatch: got {c.shape[0]} expected {self.weights.shape[0]}"
            )

        denom = self.temperature if abs(self.temperature) > 1.0e-8 else 1.0e-8
        logit = (self.bias + float(np.dot(self.weights, c))) / denom
        w = _sigmoid(logit)

        if self.ema_smooth_alpha > 0.0:
            if self._w_prev is None:
                self._w_prev = w
            else:
                alpha = self.ema_smooth_alpha
                w = float(alpha * w + (1.0 - alpha) * float(self._w_prev))
                self._w_prev = w

        return float(np.clip(w, 0.0, 1.0))


class FixedWeight:
    """Ablation helper: return a fixed alignment weight."""

    def __init__(self, weight: float):
        self.weight = float(np.clip(weight, 0.0, 1.0))

    def reset(self) -> None:  # noqa: D401
        """No-op."""

    def predict(self, c_t: np.ndarray) -> float:  # noqa: ARG002
        return self.weight

