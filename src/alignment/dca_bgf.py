from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.alignment.euclidean import apply_alignment
from src.features.covariance import compute_covariances
from src.utils.monitoring import confidence as confidence_fn
from src.utils.monitoring import entropy as entropy_fn


class DCABGF:
    """Dynamic Conditional Alignment with Behavior-Guided Feedback (MVP).

    Works in CSP feature space:
      f_final = (1-w) * f_orig + w * f_align
    where f_align is computed by applying an alignment matrix A in channel space
    (e.g. RA) before CSP transformation.
    """

    def __init__(
        self,
        csp: Any,
        lda: Any,
        alignment_matrix: np.ndarray,
        context_computer: Any,
        conditional_weight: Any,
        behavior_feedback: Any,
        use_feedback: bool = True,
    ):
        self.csp = csp
        self.lda = lda
        self.alignment_matrix = np.asarray(alignment_matrix, dtype=np.float64)
        self.context_computer = context_computer
        self.conditional_weight = conditional_weight
        self.behavior_feedback = behavior_feedback
        self.use_feedback = bool(use_feedback)

    def reset_state(self) -> None:
        if hasattr(self.context_computer, "reset"):
            self.context_computer.reset()
        if hasattr(self.conditional_weight, "reset"):
            self.conditional_weight.reset()
        if hasattr(self.behavior_feedback, "reset"):
            self.behavior_feedback.reset()

    def predict_online(
        self,
        X_target_test: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        return_details: bool = True,
        return_features: bool = False,
    ):
        self.reset_state()

        X_target_test = np.asarray(X_target_test)
        n_trials = int(X_target_test.shape[0])

        predictions: List[int] = []
        history: List[Dict] = []

        details: Dict[str, List] = {
            "w": [],
            "w_pred": [],
            "conf": [],
            "entropy": [],
            "context": [],
        }
        if return_features:
            details["features_final"] = []
        if y_true is not None:
            y_true = np.asarray(y_true, dtype=np.int64)
            details["correct"] = []

        for t in range(n_trials):
            x_t = X_target_test[t]

            f_orig = self.csp.transform(x_t[None, ...])[0]
            x_align = apply_alignment(x_t[None, ...], self.alignment_matrix)[0]
            f_align = self.csp.transform(x_align[None, ...])[0]

            trial_cov = None
            if getattr(self.context_computer, "requires_trial_covariance", False):
                cov_eps = float(getattr(self.context_computer, "cov_eps", 1.0e-6))
                trial_cov = compute_covariances(x_t[None, ...], eps=cov_eps)[0]

            c_t = self.context_computer.compute(f_orig, history, trial_cov=trial_cov)
            w_pred = float(self.conditional_weight.predict(c_t))

            if self.use_feedback:
                w = float(self.behavior_feedback.adjust_weight(w_pred, history))
            else:
                w = w_pred

            f_final = (1.0 - w) * f_orig + w * f_align

            proba = self.lda.predict_proba(f_final[None, ...])[0]
            y_hat = int(np.argmax(proba))

            conf = confidence_fn(proba)
            ent = entropy_fn(proba)

            predictions.append(y_hat)

            history.append(
                {
                    "x": f_orig,
                    "y_pred": y_hat,
                        "conf": conf,
                        "entropy": ent,
                        "w": w,
                        "w_pred": w_pred,
                        "cov": trial_cov,
                    }
                )

            if return_details:
                details["w"].append(w)
                details["w_pred"].append(w_pred)
                details["conf"].append(conf)
                details["entropy"].append(ent)
                details["context"].append(np.asarray(c_t, dtype=np.float64))
                if return_features:
                    details["features_final"].append(np.asarray(f_final, dtype=np.float64))
                if y_true is not None:
                    details["correct"].append(int(y_hat == int(y_true[t])))

        y_pred = np.asarray(predictions, dtype=np.int64)
        if not return_details:
            return y_pred

        out_details: Dict[str, np.ndarray] = {}
        for k, v in details.items():
            if k == "context":
                out_details[k] = np.stack(v, axis=0) if v else np.zeros((0, 0))
            elif k == "features_final":
                out_details[k] = np.stack(v, axis=0) if v else np.zeros((0, 0))
            else:
                out_details[k] = np.asarray(v)
        return y_pred, out_details
