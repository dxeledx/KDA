from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from pyriemann.utils.distance import distance_riemann


DEFAULT_CONTEXT_FEATURES = ("d_src", "d_tgt", "sigma_recent")
SUPPORTED_CONTEXT_FEATURES = {"d_src", "d_tgt", "sigma_recent", "d_geo"}


class ContextComputer:
    """Compute per-trial context vector c_t from CSP features.

    Supported context features:
      - d_src: ||x_t - mu_src||_2
      - d_tgt: ||x_t - mu_tgt||_2 (mu_tgt estimated online via EMA)
      - sigma_recent: std of recent feature stream
      - d_geo: Riemannian distance between per-trial covariance and source reference covariance
    """

    def __init__(
        self,
        source_mean: np.ndarray,
        context_dim: int = 3,
        feature_names: Optional[Sequence[str]] = None,
        ema_alpha: float = 0.1,
        recent_window: int = 5,
        normalize: bool = True,
        source_covariance: Optional[np.ndarray] = None,
        cov_eps: float = 1.0e-6,
    ):
        self.source_mean = np.asarray(source_mean, dtype=np.float64).ravel()
        self.feature_names = self._resolve_feature_names(
            feature_names=feature_names, context_dim=int(context_dim)
        )
        self.context_dim = len(self.feature_names)

        self.ema_alpha = float(ema_alpha)
        self.recent_window = int(recent_window)
        self.normalize = bool(normalize)
        self.cov_eps = float(cov_eps)

        self.source_covariance: Optional[np.ndarray] = None
        if source_covariance is not None:
            self.source_covariance = self._regularize_covariance(source_covariance)
        if self.requires_trial_covariance and self.source_covariance is None:
            raise ValueError("source_covariance is required when using d_geo in context.")

        self.target_mean: Optional[np.ndarray] = None
        self.stats: Optional[Dict[str, np.ndarray]] = None

    @property
    def requires_trial_covariance(self) -> bool:
        return "d_geo" in self.feature_names

    def reset(self) -> None:
        self.target_mean = None

    def fit_normalizer(
        self,
        feature_stream: np.ndarray,
        covariance_stream: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Fit mean/std stats for z-score normalization of context features."""
        feature_stream = np.asarray(feature_stream, dtype=np.float64)
        covariances = self._prepare_covariance_stream(covariance_stream, len(feature_stream))
        history: List[Dict] = []

        self.reset()
        contexts = []
        for idx, x_t in enumerate(feature_stream):
            trial_cov = None if covariances is None else covariances[idx]
            c_t = self._compute_raw(x_t, history, trial_cov=trial_cov)
            contexts.append(c_t)
            history.append({"x": np.asarray(x_t, dtype=np.float64).ravel()})

        if not contexts:
            raise ValueError("Empty feature_stream for context normalizer.")

        context_arr = np.stack(contexts, axis=0)
        stats = {
            "mean": context_arr.mean(axis=0),
            "std": context_arr.std(axis=0),
        }
        self.stats = stats
        self.reset()
        return stats

    def compute(
        self,
        x_t: np.ndarray,
        history: List[Dict],
        trial_cov: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        c_t = self._compute_raw(x_t, history, trial_cov=trial_cov)
        if self.normalize and self.stats is not None:
            return self.normalize_context(c_t)
        return c_t

    def normalize_context(self, c_t: np.ndarray) -> np.ndarray:
        if self.stats is None:
            return np.asarray(c_t, dtype=np.float64)
        mean = np.asarray(self.stats["mean"], dtype=np.float64)
        std = np.asarray(self.stats["std"], dtype=np.float64)
        return (np.asarray(c_t, dtype=np.float64) - mean) / (std + 1.0e-8)

    def _compute_raw(
        self,
        x_t: np.ndarray,
        history: List[Dict],
        trial_cov: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=np.float64).ravel()
        feature_map: Dict[str, float] = {}

        if "d_src" in self.feature_names:
            feature_map["d_src"] = float(np.linalg.norm(x_t - self.source_mean))

        if "d_tgt" in self.feature_names:
            if self.target_mean is None:
                feature_map["d_tgt"] = 0.0
            else:
                feature_map["d_tgt"] = float(np.linalg.norm(x_t - self.target_mean))

        if "sigma_recent" in self.feature_names:
            feature_map["sigma_recent"] = self._compute_sigma_recent(history)

        if "d_geo" in self.feature_names:
            if trial_cov is None:
                raise ValueError("trial_cov is required when using d_geo in context.")
            trial_cov = self._regularize_covariance(trial_cov)
            assert self.source_covariance is not None
            feature_map["d_geo"] = float(distance_riemann(trial_cov, self.source_covariance))

        c_t = np.array([feature_map[name] for name in self.feature_names], dtype=np.float64)
        self._update_target_mean(x_t)
        return c_t

    def _compute_sigma_recent(self, history: List[Dict]) -> float:
        sigma_recent = 0.0
        if len(history) < self.recent_window:
            return sigma_recent

        recent = []
        for h in history[-self.recent_window :]:
            if "x" not in h:
                continue
            recent.append(np.asarray(h["x"], dtype=np.float64).ravel())
        if len(recent) >= self.recent_window:
            sigma_recent = float(np.std(np.stack(recent, axis=0)))
        return sigma_recent

    def _prepare_covariance_stream(
        self,
        covariance_stream: Optional[np.ndarray],
        expected_length: int,
    ) -> Optional[np.ndarray]:
        if not self.requires_trial_covariance:
            return None
        if covariance_stream is None:
            raise ValueError("covariance_stream is required when using d_geo in context.")

        covariances = np.asarray(covariance_stream, dtype=np.float64)
        if covariances.shape[0] != expected_length:
            raise ValueError(
                f"covariance_stream length mismatch: got {covariances.shape[0]} expected {expected_length}"
            )
        return np.asarray(
            [self._regularize_covariance(covariance) for covariance in covariances],
            dtype=np.float64,
        )

    def _update_target_mean(self, x_t: np.ndarray) -> None:
        if self.target_mean is None:
            self.target_mean = np.asarray(x_t, dtype=np.float64).copy()
            return
        alpha = self.ema_alpha
        self.target_mean = (1.0 - alpha) * self.target_mean + alpha * np.asarray(
            x_t, dtype=np.float64
        )

    def _regularize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=np.float64)
        covariance = 0.5 * (covariance + covariance.T)
        if self.cov_eps > 0.0:
            covariance = covariance + self.cov_eps * np.eye(covariance.shape[0], dtype=np.float64)
        return covariance

    def _resolve_feature_names(
        self,
        feature_names: Optional[Sequence[str]],
        context_dim: int,
    ) -> List[str]:
        if feature_names is None:
            if context_dim == 3:
                return list(DEFAULT_CONTEXT_FEATURES)
            if context_dim == 4:
                return [*DEFAULT_CONTEXT_FEATURES, "d_geo"]
            raise ValueError(f"Unsupported context_dim: {context_dim}")

        resolved = [str(name) for name in feature_names]
        invalid = [name for name in resolved if name not in SUPPORTED_CONTEXT_FEATURES]
        if invalid:
            raise ValueError(f"Unsupported context features: {invalid}")
        return resolved
