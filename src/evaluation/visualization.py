from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _subject_labels(n: int) -> List[str]:
    return [f"A{i+1:02d}" for i in range(n)]


def plot_transfer_matrix(
    matrix: np.ndarray,
    save_path: Union[str, Path],
    title: str,
    vmin: float = 0.25,
    vmax: float = 0.85,
    annot: bool = True,
):
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    labels = _subject_labels(n)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Accuracy"},
    )
    plt.xlabel("Target Subject", fontsize=12)
    plt.ylabel("Source Subject", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scatter(
    rep_values: np.ndarray,
    beh_values: np.ndarray,
    save_path: Union[str, Path],
    title: str,
    r: Optional[float] = None,
    p_value: Optional[float] = None,
):
    rep_values = np.asarray(rep_values, dtype=float)
    beh_values = np.asarray(beh_values, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.scatter(rep_values, beh_values, alpha=0.6, s=60, c="steelblue")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
    plt.xlabel("Representation Similarity (CKA)", fontsize=12)
    plt.ylabel("Behavior (Transfer Accuracy)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if r is not None and p_value is not None:
        plt.text(
            0.02,
            0.98,
            f"r={r:.3f}, p={p_value:.3g}",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_comparison(
    methods: List[str],
    correlations: List[float],
    save_path: Union[str, Path],
    title: str = "Representation-Behavior Correlation",
):
    plt.figure(figsize=(8, 6))
    plt.bar(methods, correlations, color=["steelblue", "orange", "green"][: len(methods)])
    plt.ylabel("Pearson r", fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    for i, r in enumerate(correlations):
        plt.text(i, r + 0.02, f"{r:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_covariance_heatmaps(
    cov_means: List[np.ndarray],
    subject_labels: List[str],
    save_path: Union[str, Path],
):
    n = len(cov_means)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, C, label in zip(axes, cov_means, subject_labels):
        sns.heatmap(
            C,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(label, fontsize=12)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
