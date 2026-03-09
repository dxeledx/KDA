from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


def _ranknorm_per_target(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float64)
    for target in range(n):
        valid_sources = [src for src in range(n) if src != target]
        values = matrix[valid_sources, target]
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        if len(order) == 1:
            ranks[order] = 1.0
        else:
            ranks[order] = np.linspace(0.0, 1.0, num=len(order), dtype=np.float64)
        for idx, src in enumerate(valid_sources):
            out[src, target] = ranks[idx]
    return out


def compute_rbid_from_pairwise(
    rep_matrix: np.ndarray,
    beh_matrix: np.ndarray,
    tail_quantile: float = 0.75,
) -> Dict[str, object]:
    rep_matrix = np.asarray(rep_matrix, dtype=np.float64)
    beh_matrix = np.asarray(beh_matrix, dtype=np.float64)
    if rep_matrix.shape != beh_matrix.shape:
        raise ValueError(f"Shape mismatch: {rep_matrix.shape} vs {beh_matrix.shape}")
    if rep_matrix.ndim != 2 or rep_matrix.shape[0] != rep_matrix.shape[1]:
        raise ValueError(f"Expected square matrices, got {rep_matrix.shape}")

    rep_rank = _ranknorm_per_target(rep_matrix)
    beh_rank = _ranknorm_per_target(beh_matrix)

    mask = ~np.eye(rep_matrix.shape[0], dtype=bool)
    diff = rep_rank - beh_rank
    diff_valid = diff[mask]
    abs_diff = np.abs(diff_valid)
    rbid = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    rbid_pos = float(np.mean(np.clip(diff_valid, 0.0, None))) if diff_valid.size else 0.0
    rbid_neg = float(np.mean(np.clip(-diff_valid, 0.0, None))) if diff_valid.size else 0.0
    if abs_diff.size:
        cutoff = float(np.quantile(abs_diff, tail_quantile))
        tail = abs_diff[abs_diff >= cutoff]
        tail_rbid = float(np.mean(tail))
    else:
        cutoff = 0.0
        tail_rbid = 0.0

    pair_rows = []
    n = rep_matrix.shape[0]
    for src in range(n):
        for tgt in range(n):
            if src == tgt:
                continue
            pair_rows.append(
                {
                    "source": src + 1,
                    "target": tgt + 1,
                    "s_rep": float(rep_matrix[src, tgt]),
                    "s_beh": float(beh_matrix[src, tgt]),
                    "s_rep_rank": float(rep_rank[src, tgt]),
                    "s_beh_rank": float(beh_rank[src, tgt]),
                    "rbid_pair": float(abs(rep_rank[src, tgt] - beh_rank[src, tgt])),
                    "rbid_pos_pair": float(max(rep_rank[src, tgt] - beh_rank[src, tgt], 0.0)),
                    "rbid_neg_pair": float(max(beh_rank[src, tgt] - rep_rank[src, tgt], 0.0)),
                }
            )

    return {
        "rbid": rbid,
        "rbid_pos": rbid_pos,
        "rbid_neg": rbid_neg,
        "tail_rbid": tail_rbid,
        "tail_cutoff": cutoff,
        "pair_df": pd.DataFrame(pair_rows),
    }


def compute_rbid_from_pair_rows(
    pair_df: pd.DataFrame,
    rep_col: str,
    beh_col: str,
    source_col: str = "source",
    target_col: str = "target",
    tail_quantile: float = 0.75,
) -> Dict[str, object]:
    subjects = sorted(set(pair_df[source_col].tolist()) | set(pair_df[target_col].tolist()))
    index = {subject: idx for idx, subject in enumerate(subjects)}
    rep = np.eye(len(subjects), dtype=np.float64)
    beh = np.eye(len(subjects), dtype=np.float64)
    for row in pair_df.itertuples(index=False):
        src = getattr(row, source_col)
        tgt = getattr(row, target_col)
        rep[index[src], index[tgt]] = float(getattr(row, rep_col))
        beh[index[src], index[tgt]] = float(getattr(row, beh_col))
    out = compute_rbid_from_pairwise(rep, beh, tail_quantile=tail_quantile)
    out["subjects"] = subjects
    return out


def summarize_local_k_rbid(
    geometry_scores: np.ndarray,
    behavior_scores: np.ndarray,
) -> Dict[str, object]:
    geometry_scores = np.asarray(geometry_scores, dtype=np.float64)
    behavior_scores = np.asarray(behavior_scores, dtype=np.float64)
    if geometry_scores.shape != behavior_scores.shape:
        raise ValueError(f"Shape mismatch: {geometry_scores.shape} vs {behavior_scores.shape}")
    if geometry_scores.ndim != 2:
        raise ValueError(f"Expected [n_windows, n_actions], got {geometry_scores.shape}")

    def _ranknorm_actions(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        if len(order) == 1:
            ranks[order] = 1.0
        else:
            ranks[order] = np.linspace(0.0, 1.0, num=len(order), dtype=np.float64)
        return ranks

    g_rank = np.stack([_ranknorm_actions(row) for row in geometry_scores], axis=0)
    b_rank = np.stack([_ranknorm_actions(row) for row in behavior_scores], axis=0)
    diff = g_rank - b_rank
    abs_diff = np.abs(diff)
    return {
        "k_rbid_per_window": abs_diff.mean(axis=1),
        "k_rbid_pos_per_window": np.clip(diff, 0.0, None).mean(axis=1),
        "k_rbid_neg_per_window": np.clip(-diff, 0.0, None).mean(axis=1),
        "g_rank": g_rank,
        "b_rank": b_rank,
        "d_action": abs_diff,
    }
