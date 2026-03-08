from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import resample

try:
    # moabb>=1.0
    from moabb.datasets import BNCI2014_001 as _BCICIV2A
except ImportError:  # pragma: no cover
    # backward compatibility
    from moabb.datasets import BNCI2014001 as _BCICIV2A
from moabb.paradigms import MotorImagery

from src.utils.config import ensure_dir
from src.utils.logger import get_logger


_LOGGER = get_logger(__name__)


def _is_train_session(session_value: str) -> bool:
    s = str(session_value).strip()
    u = s.upper()
    if "TRAIN" in u:
        return True
    # Match tokens like "T", "session_T", "..._T", "..._T_..."
    return re.search(r"(^|_)T($|_)", u) is not None


def _is_test_session(session_value: str) -> bool:
    s = str(session_value).strip()
    u = s.upper()
    if "TEST" in u:
        return True
    # Match tokens like "E", "session_E", "..._E", "..._E_..."
    return re.search(r"(^|_)E($|_)", u) is not None


def _encode_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        # Map to 0..K-1 if needed
        unique = np.unique(y)
        mapping = {int(v): i for i, v in enumerate(sorted(unique.tolist()))}
        return np.array([mapping[int(v)] for v in y], dtype=np.int64)

    # Common MOABB names for BNCI2014001
    label_map = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3,
    }
    y_str = y.astype(str)
    if set(y_str).issubset(label_map.keys()):
        return np.array([label_map[v] for v in y_str], dtype=np.int64)

    unique = sorted(set(y_str.tolist()))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[v] for v in y_str], dtype=np.int64)


class BCIDataLoader:
    """BCI Competition IV Dataset 2a (BNCI2014001) loader with caching."""

    def __init__(
        self,
        processed_dir: Union[str, os.PathLike],
        subjects: List[int],
        channels: List[str],
        fmin: float,
        fmax: float,
        tmin: float,
        tmax: float,
        baseline: Optional[Tuple[float, float]] = None,
        target_sfreq: float = 250.0,
        cue_offset_s: float = 2.0,
    ):
        self.processed_dir = ensure_dir(processed_dir)
        self.subjects = subjects
        self.channels = channels
        self.fmin = fmin
        self.fmax = fmax
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.target_sfreq = float(target_sfreq)
        self.cue_offset_s = float(cue_offset_s)

        self._dataset = _BCICIV2A()
        # MOABB epochs are time-locked to cue onset (t=2s in BCI IV 2a).
        # Our docs specify absolute window [tmin, tmax] in trial time,
        # so we convert it to cue-relative for MOABB.
        moabb_tmin = self.tmin - self.cue_offset_s
        moabb_tmax = self.tmax - self.cue_offset_s
        self._paradigm = MotorImagery(
            n_classes=4,
            channels=self.channels,
            fmin=self.fmin,
            fmax=self.fmax,
            tmin=moabb_tmin,
            tmax=moabb_tmax,
            baseline=self.baseline,
        )

    @classmethod
    def from_config(cls, data_config: Dict[str, Any]) -> "BCIDataLoader":
        dataset_cfg = data_config["dataset"]
        pre_cfg = data_config["preprocessing"]
        out_cfg = data_config["output"]

        return cls(
            processed_dir=out_cfg["processed_dir"],
            subjects=list(dataset_cfg["subjects"]),
            channels=list(pre_cfg["channels"]),
            fmin=float(pre_cfg["filter"]["l_freq"]),
            fmax=float(pre_cfg["filter"]["h_freq"]),
            tmin=float(pre_cfg["epoch"]["tmin"]),
            tmax=float(pre_cfg["epoch"]["tmax"]),
            baseline=pre_cfg["epoch"].get("baseline"),
        )

    def _cache_paths(self, subject_id: int) -> Tuple[Path, Path]:
        train_path = self.processed_dir / f"A{subject_id:02d}_train.npz"
        test_path = self.processed_dir / f"A{subject_id:02d}_test.npz"
        return train_path, test_path

    def _save_npz(self, path: Path, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        np.savez(
            path,
            X=X,
            y=y,
            sfreq=self.target_sfreq,
            channels=np.asarray(self.channels),
            tmin=float(self.tmin),
            tmax=float(self.tmax),
            fmin=float(self.fmin),
            fmax=float(self.fmax),
        )

    def _load_npz(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=True) as f:
            X = f["X"].astype(np.float32)
            y = f["y"].astype(np.int64)
        return X, y

    def ensure_subject_cached(self, subject_id: int) -> None:
        train_path, test_path = self._cache_paths(subject_id)
        desired_n_samples = int(round((self.tmax - self.tmin) * self.target_sfreq))
        desired_n_channels = len(self.channels)
        if train_path.exists() and test_path.exists():
            try:
                X_train, _y_train = self._load_npz(train_path)
                X_test, _y_test = self._load_npz(test_path)
                if (
                    X_train.ndim == 3
                    and X_test.ndim == 3
                    and X_train.shape[1] == desired_n_channels
                    and X_test.shape[1] == desired_n_channels
                    and X_train.shape[2] == desired_n_samples
                    and X_test.shape[2] == desired_n_samples
                    and X_train.shape[0] == X_test.shape[0]
                ):
                    return
                _LOGGER.warning(
                    "Cache shape mismatch for subject %s; regenerating (train=%s test=%s)",
                    subject_id,
                    tuple(X_train.shape),
                    tuple(X_test.shape),
                )
            except Exception:
                _LOGGER.warning("Failed to load cache for subject %s; regenerating", subject_id)

        _LOGGER.info("Fetching subject %s via MOABB...", subject_id)
        X, y, meta = self._paradigm.get_data(self._dataset, subjects=[subject_id])

        y = _encode_labels(np.asarray(y))

        if not isinstance(meta, pd.DataFrame):
            meta = pd.DataFrame(meta)

        if "session" not in meta.columns:
            raise RuntimeError(f"MOABB meta missing 'session' column: {meta.columns}")

        session_series = meta["session"].astype(str)
        train_mask = session_series.map(_is_train_session).to_numpy(dtype=bool)
        test_mask = session_series.map(_is_test_session).to_numpy(dtype=bool)

        if train_mask.any() and test_mask.any():
            pass
        else:
            unique_sessions = sorted(pd.unique(session_series).tolist())
            if len(unique_sessions) < 2:
                raise RuntimeError(
                    f"Unable to infer train/test sessions from: {unique_sessions}"
                )
            train_value, test_value = unique_sessions[0], unique_sessions[1]
            train_mask = (session_series == train_value).to_numpy(dtype=bool)
            test_mask = (session_series == test_value).to_numpy(dtype=bool)

        X_train, y_train = np.asarray(X)[train_mask], y[train_mask]
        X_test, y_test = np.asarray(X)[test_mask], y[test_mask]

        # Resample if needed
        if X_train.shape[2] != desired_n_samples:
            _LOGGER.warning(
                "Resampling subject %s: %s -> %s samples",
                subject_id,
                X_train.shape[2],
                desired_n_samples,
            )
            X_train = resample(X_train, desired_n_samples, axis=2)
        if X_test.shape[2] != desired_n_samples:
            X_test = resample(X_test, desired_n_samples, axis=2)

        _LOGGER.info(
            "Subject %s shapes: train=%s test=%s",
            subject_id,
            tuple(X_train.shape),
            tuple(X_test.shape),
        )

        self._save_npz(train_path, X_train, y_train)
        self._save_npz(test_path, X_test, y_test)

    def load_subject(self, subject_id: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        self.ensure_subject_cached(subject_id)
        train_path, test_path = self._cache_paths(subject_id)
        if split == "train":
            return self._load_npz(train_path)
        if split == "test":
            return self._load_npz(test_path)
        raise ValueError(f"Unknown split: {split}")

    def get_train_test_split(
        self, subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, y_train = self.load_subject(subject_id, split="train")
        X_test, y_test = self.load_subject(subject_id, split="test")
        return X_train, y_train, X_test, y_test
