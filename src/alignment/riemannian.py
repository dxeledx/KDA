from __future__ import annotations

from typing import Optional

import numpy as np
from pyriemann.utils.mean import mean_riemann

from src.alignment.euclidean import compute_alignment_matrix


class RiemannianAlignment:
    def __init__(self, eps: float = 1.0e-6):
        self.eps = float(eps)
        self.C_ref: Optional[np.ndarray] = None

    def fit(self, C_source: np.ndarray):  # noqa: N803
        self.C_ref = mean_riemann(C_source)
        return self

    def compute_matrix(self, C_target: np.ndarray) -> np.ndarray:  # noqa: N803
        if self.C_ref is None:
            raise RuntimeError("RiemannianAlignment is not fitted.")
        C_target_mean = mean_riemann(C_target)
        return compute_alignment_matrix(self.C_ref, C_target_mean, eps=self.eps)
