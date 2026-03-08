#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.kcar_policy import (
    attach_policy_subject_metrics,
    build_budgeted_policy_benchmark,
    compare_policies_against_baseline,
    summarize_budget_curves,
)
from src.utils.config import ensure_dir
from src.utils.logger import get_logger


logger = get_logger("kcar_safe_policy")


def _save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_policy_switches(
    policy_windows: pd.DataFrame,
    out_path: Path,
    policy: str,
    setting: str,
    coverage: float,
) -> None:
    subset = policy_windows[
        (policy_windows["policy"] == policy)
        & (policy_windows["setting"] == setting)
        & np.isclose(policy_windows["coverage"], float(coverage))
    ].copy()
    subjects = sorted(subset["subject"].astype(int).unique().tolist())
    ncols = 3
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)

    for axis in axes.flat:
        axis.axis("off")

    for axis, subject in zip(axes.flat, subjects):
        axis.axis("on")
        subject_df = subset[subset["subject"] == subject].sort_values("window_id")
        x_axis = subject_df["window_id"].to_numpy(dtype=np.int64)
        actions = (subject_df["selected_action"] == "use_partial_alignment").astype(np.float64)
        axis.step(x_axis, actions, where="mid", linewidth=1.8, label="use_partial_alignment")
        axis.plot(
            x_axis,
            subject_df["selected_delta_vs_ra"],
            color="tab:blue",
            linewidth=1.5,
            label="delta_vs_ra",
        )
        axis.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        axis.set_title(f"Target A{subject:02d}")
        axis.set_xlabel("Window")
        axis.set_ylabel("Action / delta")
        axis.grid(alpha=0.3)

    fig.suptitle(f"{policy} | {setting} | coverage={coverage:.1f}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_coverage_curve(
    subject_summary_df: pd.DataFrame,
    value_column: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7.0, 4.8))
    for (policy, setting), frame in subject_summary_df.groupby(["policy", "setting"]):
        if policy == "ra":
            continue
        curve = (
            frame.groupby("coverage")[value_column]
            .mean()
            .reset_index()
            .sort_values("coverage")
        )
        plt.plot(
            curve["coverage"].to_numpy(dtype=np.float64),
            curve[value_column].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.6,
            label=f"{policy}-{setting}",
        )
    plt.xlabel("Deviation coverage")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_budget_curve_frame(
    budget_curve: pd.DataFrame,
    value_column: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7.0, 4.8))
    for (policy, setting), frame in budget_curve.groupby(["policy", "setting"]):
        if policy == "ra":
            continue
        ordered = frame.sort_values("coverage")
        plt.plot(
            ordered["coverage"].to_numpy(dtype=np.float64),
            ordered[value_column].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.6,
            label=f"{policy}-{setting}",
        )
    plt.xlabel("Deviation coverage")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_retrospective_vs_causal_gap(
    comparison_df: pd.DataFrame,
    out_path: Path,
) -> None:
    subset = comparison_df[comparison_df["baseline_policy"] == "ra"].copy()
    plt.figure(figsize=(7.0, 4.8))
    for policy, frame in subset.groupby("policy"):
        pivot = frame.pivot(index="coverage", columns="setting", values="mean_delta_vs_baseline")
        if {"retrospective", "near_causal"}.issubset(pivot.columns):
            gap = pivot["retrospective"] - pivot["near_causal"]
            plt.plot(
                gap.index.to_numpy(dtype=np.float64),
                gap.to_numpy(dtype=np.float64),
                marker="o",
                linewidth=1.6,
                label=str(policy),
            )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("Deviation coverage")
    plt.ylabel("Retrospective - near-causal delta")
    plt.title("Retrospective vs near-causal gap")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-dir",
        default="results/stage3/kcar_diagnostic/latest",
        help="Directory containing window_metrics.csv and summary.json",
    )
    parser.add_argument("--run-name", default="latest")
    parser.add_argument(
        "--coverages",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma separated coverage budgets",
    )
    parser.add_argument("--force", action="store_true", help="Run policy even if diagnostic gate fails")
    args = parser.parse_args()

    diagnostic_dir = Path(args.diagnostic_dir)
    window_metrics_path = diagnostic_dir / "window_metrics.csv"
    summary_path = diagnostic_dir / "summary.json"
    details_dir = diagnostic_dir / "details"

    if not window_metrics_path.exists():
        raise FileNotFoundError(f"Missing diagnostic results: {window_metrics_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing diagnostic summary: {summary_path}")

    diagnostic_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mean_auroc = float(diagnostic_summary.get("mean_auroc", float("nan")))
    if not args.force and (not np.isfinite(mean_auroc) or mean_auroc < 0.60):
        raise RuntimeError(
            f"Diagnostic gate not met (mean_auroc={mean_auroc:.4f}); use --force to override."
        )

    coverages = [float(token.strip()) for token in args.coverages.split(",") if token.strip()]
    window_metrics = pd.read_csv(window_metrics_path)
    output_dir = ensure_dir(f"results/stage3/kcar_policy/{args.run_name}")

    policy_windows = build_budgeted_policy_benchmark(
        window_metrics,
        policies={"kcar": "rho_kcar", "d_tgt": "d_tgt", "sigma_recent": "sigma_recent"},
        settings=("retrospective", "near_causal"),
        coverages=coverages,
    )
    policy_windows.to_csv(output_dir / "policy_windows.csv", index=False)

    subject_metrics = attach_policy_subject_metrics(policy_windows, details_dir=details_dir)
    subject_summary_df, summary = summarize_budget_curves(subject_metrics)
    subject_summary_df.to_csv(output_dir / "policy_subjects.csv", index=False)

    budget_curve = (
        subject_summary_df.groupby(["policy", "setting", "coverage"], as_index=False)[
            ["policy_accuracy", "balanced_accuracy", "macro_f1", "late_session_acc", "delta_vs_ra"]
        ]
        .mean()
        .rename(
            columns={
                "policy_accuracy": "mean_acc",
                "balanced_accuracy": "mean_bacc",
                "macro_f1": "mean_macro_f1",
            }
        )
    )
    worst_curve = (
        subject_summary_df.groupby(["policy", "setting", "coverage"], as_index=False)["delta_vs_ra"]
        .min()
        .rename(columns={"delta_vs_ra": "worst_subject_delta"})
    )
    budget_curve = budget_curve.merge(
        worst_curve, on=["policy", "setting", "coverage"], how="left"
    ).sort_values(["policy", "setting", "coverage"])
    budget_curve.to_csv(output_dir / "budget_curve.csv", index=False)

    comparison_ra = compare_policies_against_baseline(subject_metrics, baseline_policy="ra")
    comparison_dtgt = compare_policies_against_baseline(
        subject_metrics[subject_metrics["policy"].isin(["kcar", "d_tgt"])],
        baseline_policy="d_tgt",
    )
    comparison_sigma = compare_policies_against_baseline(
        subject_metrics[subject_metrics["policy"].isin(["kcar", "sigma_recent"])],
        baseline_policy="sigma_recent",
    )
    comparison_df = pd.concat(
        [comparison_ra, comparison_dtgt, comparison_sigma], axis=0, ignore_index=True
    )
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)

    policy_summary_rows = []
    for policy, settings_summary in summary.items():
        for setting, coverage_summary in settings_summary.items():
            for coverage_key, values in coverage_summary.items():
                policy_summary_rows.append(
                    {
                        "policy": policy,
                        "setting": setting,
                        "coverage": float(coverage_key),
                        **values,
                    }
                )
    policy_summary_df = pd.DataFrame(policy_summary_rows).sort_values(
        ["policy", "setting", "coverage"]
    )
    policy_summary_df.to_csv(output_dir / "policy_summary.csv", index=False)
    _save_json(
        {
            "diagnostic_dir": str(diagnostic_dir),
            "coverages": coverages,
            "summary": summary,
        },
        output_dir / "summary.json",
    )

    default_coverage = 0.5 if 0.5 in coverages else float(
        coverages[min(len(coverages) - 1, len(coverages) // 2)]
    )
    _plot_policy_switches(
        policy_windows,
        output_dir / "policy_vs_ra_window_switches.pdf",
        policy="kcar",
        setting="retrospective",
        coverage=default_coverage,
    )
    _plot_coverage_curve(
        subject_summary_df,
        value_column="policy_accuracy",
        out_path=output_dir / "coverage_performance_curve.pdf",
        title="Coverage vs mean accuracy",
        ylabel="Mean accuracy",
    )
    _plot_budget_curve_frame(
        budget_curve,
        value_column="worst_subject_delta",
        out_path=output_dir / "coverage_worst_subject_curve.pdf",
        title="Coverage vs worst-subject delta",
        ylabel="Worst-subject delta vs RA",
    )
    _plot_retrospective_vs_causal_gap(
        comparison_df,
        output_dir / "retrospective_vs_causal_gap.pdf",
    )
    logger.info("Saved KCAR policy outputs to %s", output_dir)


if __name__ == "__main__":
    main()
