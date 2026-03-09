#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ensure_dir
from src.utils.logger import get_logger


logger = get_logger("e0_refresh_baselines")

CLASSICAL_METHODS = {
    "noalign": "No Alignment",
    "ea": "EA",
    "ra": "RA",
}
KOOPMAN_METHODS = [
    "Koopman-noalign",
    "Static Koopman aligner",
]
MAIN_METHOD_ORDER = [
    "No Alignment",
    "EA",
    "RA",
    "Koopman-noalign-r48",
    "Static Koopman aligner-r48",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="2026-03-09-e0-r1")
    parser.add_argument("--sync-figures-dir", default="results/figures")
    parser.add_argument("--memo-path", default="docs/KSDA/25-e0-baseline-refresh-memo.md")
    return parser


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_python(script: Path, *args: str) -> None:
    cmd = [sys.executable, str(script), *args]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_e0_main_table(
    classical_root: Path,
    phenomenon_root: Path,
    koopman_rank: int,
) -> pd.DataFrame:
    classical_rows = []
    for method_key, method_name in CLASSICAL_METHODS.items():
        summary = _read_json(classical_root / method_key / "summary.json")
        classical_rows.append(
            {
                "method": method_name,
                "accuracy_mean": float(summary["accuracy"]["mean"]),
                "accuracy_std": float(summary["accuracy"]["std"]),
                "kappa_mean": float(summary["kappa"]["mean"]),
                "kappa_std": float(summary["kappa"]["std"]),
            }
        )

    loso_df = pd.read_csv(phenomenon_root / "loso_method_summary.csv")
    rbid_df = pd.read_csv(phenomenon_root / "rbid_method_comparison.csv")

    koopman_rows = []
    for base_name in KOOPMAN_METHODS:
        loso_row = loso_df.loc[loso_df["method"] == base_name].iloc[0]
        rbid_row = rbid_df.loc[rbid_df["method"] == base_name].iloc[0]
        koopman_rows.append(
            {
                "method": f"{base_name}-r{koopman_rank}",
                "accuracy_mean": float(loso_row["accuracy_mean"]),
                "accuracy_std": float(loso_row["accuracy_std"]),
                "kappa_mean": float(loso_row["kappa_mean"]),
                "kappa_std": float(loso_row["kappa_std"]),
                "rbid": float(rbid_row["rbid"]),
                "rbid_pos": float(rbid_row["rbid_pos"]),
                "rbid_neg": float(rbid_row["rbid_neg"]),
                "tail_rbid": float(rbid_row["tail_rbid"]),
                "pearson_r": float(rbid_row["pearson_r"]),
            }
        )

    main_df = pd.DataFrame(classical_rows + koopman_rows)
    rbid_map = {
        row["method"]: row
        for row in rbid_df.to_dict(orient="records")
        if row["method"] in CLASSICAL_METHODS.values()
    }
    for idx, row in main_df.iterrows():
        if row["method"] in rbid_map:
            entry = rbid_map[row["method"]]
            main_df.loc[idx, "rbid"] = float(entry["rbid"])
            main_df.loc[idx, "rbid_pos"] = float(entry["rbid_pos"])
            main_df.loc[idx, "rbid_neg"] = float(entry["rbid_neg"])
            main_df.loc[idx, "tail_rbid"] = float(entry["tail_rbid"])
            main_df.loc[idx, "pearson_r"] = float(entry["pearson_r"])

    main_df = main_df.set_index("method").loc[MAIN_METHOD_ORDER].reset_index()
    return main_df[
        [
            "method",
            "accuracy_mean",
            "accuracy_std",
            "kappa_mean",
            "kappa_std",
            "rbid",
            "rbid_pos",
            "rbid_neg",
            "tail_rbid",
            "pearson_r",
        ]
    ]


def build_historical_koopman_table(
    r16_root: Path,
    r48_root: Path,
) -> pd.DataFrame:
    def _subset(root: Path, suffix: str) -> pd.DataFrame:
        loso_df = pd.read_csv(root / "loso_method_summary.csv")
        rbid_df = pd.read_csv(root / "rbid_method_comparison.csv")
        loso_df = loso_df.loc[loso_df["method"].isin(KOOPMAN_METHODS)][
            ["method", "accuracy_mean", "accuracy_std", "kappa_mean", "kappa_std"]
        ]
        rbid_df = rbid_df.loc[rbid_df["method"].isin(KOOPMAN_METHODS)][
            ["method", "pearson_r", "rbid", "tail_rbid"]
        ]
        merged = loso_df.merge(rbid_df, on="method", how="inner")
        return merged.rename(columns={col: f"{col}_{suffix}" for col in merged.columns if col != "method"})

    hist = _subset(r16_root, "r16").merge(_subset(r48_root, "r48"), on="method", how="inner")
    return hist


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(headers[idx]), max((len(row[idx]) for row in rows), default=0))
        for idx in range(len(headers))
    ]

    def _fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def render_e0_memo(
    run_dir: Path,
    main_table: pd.DataFrame,
    historical_table: pd.DataFrame,
) -> str:
    best_row = main_table.sort_values("accuracy_mean", ascending=False).iloc[0]
    lowest_rbid_row = main_table.sort_values("rbid", ascending=True).iloc[0]
    return (
        "# E0 Baseline Refresh Memo\n\n"
        f"**Run dir**: `{run_dir}`  \n"
        "**Protocol**: BNCI2014001 LOSO + pairwise transfer mismatch refresh  \n"
        "**Rank policy**: `r48` as main Koopman baseline, `r16` as historical appendix\n\n"
        "## Main table\n\n"
        f"{_markdown_table(main_table)}\n\n"
        "## Historical Koopman comparison\n\n"
        f"{_markdown_table(historical_table)}\n\n"
        "## Key findings\n\n"
        f"- Highest LOSO mean accuracy in the refreshed main table: `{best_row['method']}` = `{best_row['accuracy_mean']:.4f}`.\n"
        f"- Lowest RBID in the refreshed main table: `{lowest_rbid_row['method']}` = `{lowest_rbid_row['rbid']:.4f}`.\n"
        "- `r48` is the paper-facing Koopman baseline used for E1 controls.\n"
        "- E1 control method is fixed to `Static Koopman aligner-r48`.\n"
    )


def _sync_summary_files(figures_dir: Path, summary_dir: Path, r48_root: Path, historical_table: pd.DataFrame) -> None:
    ensure_dir(figures_dir)
    shutil.copy2(summary_dir / "e0_main_table.csv", figures_dir / "e0_main_table.csv")
    shutil.copy2(r48_root / "rbid_method_comparison.csv", figures_dir / "rbid_method_comparison.csv")
    shutil.copy2(r48_root / "rbid_summary.csv", figures_dir / "rbid_summary.csv")
    shutil.copy2(r48_root / "pairwise_scores.csv", figures_dir / "pairwise_scores.csv")
    historical_table.to_csv(figures_dir / "historical_koopman_comparison.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run_dir = ensure_dir(Path("results/e0") / args.run_name)
    baseline_root = ensure_dir(run_dir / "baselines")
    phenomenon_root = ensure_dir(run_dir / "phenomenon")
    summary_root = ensure_dir(run_dir / "summary")
    classical_fig_root = ensure_dir(run_dir / "figures" / "classical")
    r48_root = ensure_dir(phenomenon_root / "r48")
    r16_root = ensure_dir(phenomenon_root / "r16")

    _run_python(PROJECT_ROOT / "experiments" / "baseline_csp_lda.py", "--out-dir", str(baseline_root / "noalign"), "--fig-dir", str(classical_fig_root))
    _run_python(PROJECT_ROOT / "experiments" / "baseline_ea.py", "--out-dir", str(baseline_root / "ea"), "--fig-dir", str(classical_fig_root))
    _run_python(PROJECT_ROOT / "experiments" / "baseline_ra.py", "--out-dir", str(baseline_root / "ra"), "--fig-dir", str(classical_fig_root))

    _run_python(
        PROJECT_ROOT / "experiments" / "phenomenon_verification.py",
        "--classical-root",
        str(baseline_root),
        "--output-root",
        str(r48_root),
        "--koopman-pca-rank",
        "48",
    )
    _run_python(
        PROJECT_ROOT / "experiments" / "phenomenon_verification.py",
        "--classical-root",
        str(baseline_root),
        "--output-root",
        str(r16_root),
        "--koopman-pca-rank",
        "16",
    )

    main_table = build_e0_main_table(baseline_root, r48_root, koopman_rank=48)
    historical_table = build_historical_koopman_table(r16_root, r48_root)
    main_table.to_csv(summary_root / "e0_main_table.csv", index=False)
    historical_table.to_csv(summary_root / "e0_historical_koopman_table.csv", index=False)

    memo_text = render_e0_memo(run_dir, main_table, historical_table)
    (summary_root / "e0_memo.md").write_text(memo_text, encoding="utf-8")
    memo_path = Path(args.memo_path)
    ensure_dir(memo_path.parent)
    memo_path.write_text(memo_text, encoding="utf-8")

    _sync_summary_files(Path(args.sync_figures_dir), summary_root, r48_root, historical_table)
    logger.info("E0 refresh complete. Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
