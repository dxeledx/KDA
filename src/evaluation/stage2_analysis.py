from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def pair_subject_deltas(
    candidate_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    metric: str = "accuracy",
) -> pd.DataFrame:
    merged = candidate_df[["target_subject", metric]].merge(
        reference_df[["target_subject", metric]],
        on="target_subject",
        suffixes=("_candidate", "_reference"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping target_subject values between candidate and reference.")

    merged["delta"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    return merged.sort_values("target_subject").reset_index(drop=True)


def summarize_against_reference(
    candidate_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    metric: str = "accuracy",
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> Dict[str, float]:
    paired = pair_subject_deltas(candidate_df, reference_df, metric=metric)
    candidate_values = paired[f"{metric}_candidate"].to_numpy(dtype=np.float64)
    reference_values = paired[f"{metric}_reference"].to_numpy(dtype=np.float64)
    deltas = paired["delta"].to_numpy(dtype=np.float64)

    ci_low, ci_high = bootstrap_mean_difference(
        deltas, n_bootstrap=n_bootstrap, random_seed=random_seed
    )

    summary = {
        "mean": float(candidate_values.mean()),
        "std": float(candidate_values.std(ddof=1)) if candidate_values.size > 1 else 0.0,
        "reference_mean": float(reference_values.mean()),
        "delta_vs_reference": float(deltas.mean()),
        "wins": int(np.sum(deltas > 0.0)),
        "losses": int(np.sum(deltas < 0.0)),
        "draws": int(np.sum(np.isclose(deltas, 0.0))),
        "p_value": paired_wilcoxon_pvalue(deltas),
        "effect_size": paired_effect_size(deltas),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
    }
    return summary


def paired_wilcoxon_pvalue(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0 or np.allclose(deltas, 0.0):
        return float("nan")
    try:
        _stat, p_value = wilcoxon(deltas, zero_method="wilcox")
    except ValueError:
        return float("nan")
    return float(p_value)


def paired_effect_size(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size < 2:
        return float("nan")
    std = float(deltas.std(ddof=1))
    if std <= 1.0e-12:
        return float("nan")
    return float(deltas.mean() / std)


def bootstrap_mean_difference(
    deltas: np.ndarray,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan"), float("nan")

    rng = np.random.RandomState(random_seed)
    samples = rng.randint(0, deltas.size, size=(n_bootstrap, deltas.size))
    boot_means = deltas[samples].mean(axis=1)
    low, high = np.percentile(boot_means, ci)
    return float(low), float(high)


def sliding_window_mean(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    window = max(1, min(int(window_size), int(values.size)))
    indices = []
    window_means = []
    for end in range(window, values.size + 1):
        start = end - window
        window_means.append(float(values[start:end].mean()))
        indices.append(end - 1)
    return np.asarray(indices, dtype=np.int64), np.asarray(window_means, dtype=np.float64)
