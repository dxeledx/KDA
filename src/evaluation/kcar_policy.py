from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


def select_threshold_for_training_windows(
    window_metrics: pd.DataFrame,
    score_column: str = "rho_kcar",
    positive_label: str = "deviation-beneficial",
    negative_label: str = "ra-safe",
) -> float:
    eligible = window_metrics[window_metrics["label"].isin([positive_label, negative_label])].copy()
    if eligible.empty:
        raise ValueError("No eligible windows for threshold selection.")

    scores = np.sort(eligible[score_column].astype(np.float64).unique())
    if scores.size == 1:
        return float(scores[0])

    candidates = ((scores[:-1] + scores[1:]) / 2.0).tolist()
    candidates.insert(0, float(scores[0]) - 1.0e-6)
    candidates.append(float(scores[-1]) + 1.0e-6)

    y_true = (eligible["label"] == positive_label).astype(np.int64).to_numpy()
    best_threshold = float(candidates[0])
    best_score = -float("inf")
    for threshold in candidates:
        y_pred = (eligible[score_column].to_numpy(dtype=np.float64) > float(threshold)).astype(np.int64)
        score = float(balanced_accuracy_score(y_true, y_pred))
        if score > best_score + 1.0e-12:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def make_near_causal_scores(
    window_metrics: pd.DataFrame,
    score_column: str,
    output_column: str | None = None,
    subject_column: str = "subject",
    order_column: str = "window_id",
) -> pd.DataFrame:
    frame = window_metrics.copy()
    out_col = output_column or score_column.replace("_retro", "_causal")
    groups = []

    for _subject, group in frame.groupby(subject_column):
        ordered = group.sort_values(order_column).copy()
        shifted = ordered[score_column].shift(1)
        if shifted.isna().any():
            shifted = shifted.fillna(ordered[score_column].iloc[0])
        ordered[out_col] = shifted.to_numpy(dtype=np.float64)
        groups.append(ordered)

    return pd.concat(groups, axis=0, ignore_index=True).sort_values(
        [subject_column, order_column]
    ).reset_index(drop=True)


def add_budget_rank_columns(
    window_metrics: pd.DataFrame,
    score_columns: Sequence[str],
    subject_column: str = "subject",
    order_column: str = "window_id",
) -> pd.DataFrame:
    frame = window_metrics.copy()
    for score_column in score_columns:
        rank_column = f"budget_rank_{score_column}"
        groups = []
        for _subject, group in frame.groupby(subject_column):
            ordered = group.sort_values(order_column).copy()
            order = np.argsort(-ordered[score_column].to_numpy(dtype=np.float64), kind="mergesort")
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1)
            ordered[rank_column] = ranks
            groups.append(ordered)
        frame = pd.concat(groups, axis=0, ignore_index=True).sort_values(
            [subject_column, order_column]
        ).reset_index(drop=True)
    return frame


def apply_ra_first_policy(
    window_metrics: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    frame = window_metrics.copy()
    use_partial = frame[score_column].to_numpy(dtype=np.float64) > float(threshold)
    frame["selected_action"] = np.where(use_partial, "use_partial_alignment", "stay_with_ra")
    frame["selected_accuracy"] = np.where(
        use_partial,
        frame["acc_w05"].to_numpy(dtype=np.float64),
        frame["acc_ra"].to_numpy(dtype=np.float64),
    )
    frame["selected_delta_vs_ra"] = frame["selected_accuracy"] - frame["acc_ra"].to_numpy(
        dtype=np.float64
    )
    return frame


def _coverage_to_budget(n_windows: int, coverage: float) -> int:
    coverage = float(coverage)
    if coverage <= 0.0:
        return 0
    budget = int(round(float(n_windows) * coverage))
    return max(1, min(int(n_windows), budget))


def _apply_budgeted_policy_for_subject(
    subject_df: pd.DataFrame,
    score_column: str,
    coverage: float,
    order_column: str = "window_id",
) -> pd.DataFrame:
    frame = subject_df.sort_values(order_column).copy()
    scores = frame[score_column].to_numpy(dtype=np.float64)
    budget = _coverage_to_budget(len(frame), coverage)
    order = np.argsort(-scores, kind="mergesort")
    use_partial = np.zeros(len(frame), dtype=bool)
    if budget > 0:
        use_partial[order[:budget]] = True

    frame["score_column"] = score_column
    frame["score_value"] = scores
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    frame["budget_rank"] = ranks
    frame["budget_k"] = budget
    frame["use_partial_alignment"] = use_partial.astype(np.int64)
    frame["selected_action"] = np.where(use_partial, "use_partial_alignment", "stay_with_ra")
    frame["selected_accuracy"] = np.where(
        use_partial,
        frame["acc_w05"].to_numpy(dtype=np.float64),
        frame["acc_ra"].to_numpy(dtype=np.float64),
    )
    frame["selected_delta_vs_ra"] = frame["selected_accuracy"] - frame["acc_ra"].to_numpy(
        dtype=np.float64
    )
    return frame


def build_budgeted_policy_benchmark(
    window_metrics: pd.DataFrame,
    policies: Mapping[str, str],
    settings: Sequence[str] = ("retrospective", "near_causal"),
    coverages: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    subject_column: str = "subject",
    order_column: str = "window_id",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    ordered = window_metrics.sort_values([subject_column, order_column]).reset_index(drop=True)

    for setting in settings:
        suffix = "retro" if setting == "retrospective" else "causal"
        for coverage in coverages:
            ra_frame = ordered.copy()
            ra_frame["policy"] = "ra"
            ra_frame["setting"] = setting
            ra_frame["coverage"] = float(coverage)
            ra_frame["score_column"] = f"ra_{suffix}"
            ra_frame["score_value"] = np.nan
            ra_frame["budget_rank"] = 0
            ra_frame["budget_k"] = 0
            ra_frame["use_partial_alignment"] = 0
            ra_frame["selected_action"] = "stay_with_ra"
            ra_frame["selected_accuracy"] = ra_frame["acc_ra"].to_numpy(dtype=np.float64)
            ra_frame["selected_delta_vs_ra"] = 0.0
            frames.append(ra_frame)

            for policy_name, score_prefix in policies.items():
                score_column = f"{score_prefix}_{suffix}"
                if score_column not in ordered.columns:
                    raise ValueError(f"Missing score column: {score_column}")
                subject_frames = []
                for _subject, subject_df in ordered.groupby(subject_column):
                    out = _apply_budgeted_policy_for_subject(
                        subject_df=subject_df,
                        score_column=score_column,
                        coverage=float(coverage),
                        order_column=order_column,
                    )
                    out["policy"] = policy_name
                    out["setting"] = setting
                    out["coverage"] = float(coverage)
                    subject_frames.append(out)
                frames.append(pd.concat(subject_frames, axis=0, ignore_index=True))

    return pd.concat(frames, axis=0, ignore_index=True)


def select_subject_thresholds(
    window_metrics: pd.DataFrame,
    score_column: str = "rho_kcar",
) -> pd.DataFrame:
    rows: List[Dict] = []
    subjects = sorted(window_metrics["subject"].astype(int).unique().tolist())
    for subject in subjects:
        train_df = window_metrics[window_metrics["subject"] != subject]
        threshold = select_threshold_for_training_windows(train_df, score_column=score_column)
        rows.append({"subject": int(subject), "threshold": float(threshold)})
    return pd.DataFrame(rows)


def summarize_budget_curves(
    policy_windows: pd.DataFrame,
    baseline_column: str = "ra_accuracy",
) -> Tuple[pd.DataFrame, Dict]:
    subject_rows: List[Dict] = []
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    group_cols = ["policy", "setting", "coverage", "subject"]
    for keys, subject_df in policy_windows.groupby(group_cols):
        policy, setting, coverage, subject = keys
        effective_baseline_column = baseline_column
        if effective_baseline_column not in subject_df.columns and "acc_ra" in subject_df.columns:
            effective_baseline_column = "acc_ra"
        if "policy_accuracy" in subject_df.columns and effective_baseline_column in subject_df.columns:
            row = {
                "policy": str(policy),
                "setting": str(setting),
                "coverage": float(coverage),
                "subject": int(subject),
                "policy_accuracy": float(subject_df["policy_accuracy"].iloc[0]),
                "ra_accuracy": float(subject_df[effective_baseline_column].iloc[0]),
                "delta_vs_ra": float(subject_df["delta_vs_ra"].iloc[0]),
                "late_session_acc": float(subject_df["late_session_acc"].iloc[0]),
                "negative_windows": int(subject_df.get("negative_windows", pd.Series([0])).iloc[0]),
            }
        else:
            baseline = subject_df[effective_baseline_column].to_numpy(dtype=np.float64)
            selected = subject_df["selected_accuracy"].to_numpy(dtype=np.float64)
            late_start = max(0, len(subject_df) - max(1, len(subject_df) // 3))
            row = {
                "policy": str(policy),
                "setting": str(setting),
                "coverage": float(coverage),
                "subject": int(subject),
                "policy_accuracy": float(selected.mean()),
                "ra_accuracy": float(baseline.mean()),
                "delta_vs_ra": float(selected.mean() - baseline.mean()),
                "late_session_acc": float(selected[late_start:].mean()),
                "negative_windows": int(np.sum(selected < baseline)),
            }
        if "balanced_accuracy" in subject_df.columns:
            row["balanced_accuracy"] = float(subject_df["balanced_accuracy"].iloc[0])
        if "macro_f1" in subject_df.columns:
            row["macro_f1"] = float(subject_df["macro_f1"].iloc[0])
        subject_rows.append(row)

    subject_summary_df = pd.DataFrame(subject_rows).sort_values(
        ["policy", "setting", "coverage", "subject"]
    ).reset_index(drop=True)

    for (policy, setting, coverage), frame in subject_summary_df.groupby(
        ["policy", "setting", "coverage"]
    ):
        policy_key = str(policy)
        setting_key = str(setting)
        coverage_key = f"{float(coverage):.1f}"
        summary.setdefault(policy_key, {}).setdefault(setting_key, {})[coverage_key] = {
            "mean_accuracy": float(frame["policy_accuracy"].mean()),
            "std_accuracy": float(frame["policy_accuracy"].std(ddof=1))
            if len(frame) > 1
            else 0.0,
            "mean_delta_vs_ra": float(frame["delta_vs_ra"].mean()),
            "wins_vs_ra": int(np.sum(frame["delta_vs_ra"] > 0.0)),
            "losses_vs_ra": int(np.sum(frame["delta_vs_ra"] < 0.0)),
            "worst_subject_delta": float(frame["delta_vs_ra"].min()),
            "late_session_acc": float(frame["late_session_acc"].mean()),
            "negative_transfer_subject_count": int(np.sum(frame["delta_vs_ra"] < 0.0)),
            "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean())
            if "balanced_accuracy" in frame.columns
            else float("nan"),
            "mean_macro_f1": float(frame["macro_f1"].mean())
            if "macro_f1" in frame.columns
            else float("nan"),
        }
    return subject_summary_df, summary


def summarize_policy_windows(
    policy_windows: pd.DataFrame,
    baseline_column: str = "acc_ra",
) -> Tuple[pd.DataFrame, Dict]:
    return summarize_budget_curves(policy_windows, baseline_column=baseline_column)


def compare_policies_against_baseline(
    policy_windows: pd.DataFrame,
    baseline_policy: str = "ra",
) -> pd.DataFrame:
    subject_summary_df, _summary = summarize_budget_curves(policy_windows)
    baseline = subject_summary_df[subject_summary_df["policy"] == baseline_policy]
    if baseline.empty:
        raise ValueError(f"Missing baseline policy rows for {baseline_policy}")

    rows: List[Dict] = []
    for (policy, setting, coverage), frame in subject_summary_df.groupby(
        ["policy", "setting", "coverage"]
    ):
        if policy == baseline_policy:
            continue
        merged = frame.merge(
            baseline[
                [
                    "subject",
                    "setting",
                    "coverage",
                    "policy_accuracy",
                    "late_session_acc",
                ]
            ],
            on=["subject", "setting", "coverage"],
            suffixes=("", "_baseline"),
            how="inner",
        )
        if merged.empty:
            continue
        delta = merged["policy_accuracy"] - merged["policy_accuracy_baseline"]
        rows.append(
            {
                "policy": str(policy),
                "setting": str(setting),
                "coverage": float(coverage),
                "baseline_policy": str(baseline_policy),
                "mean_accuracy": float(merged["policy_accuracy"].mean()),
                "mean_delta_vs_baseline": float(delta.mean()),
                "wins": int(np.sum(delta > 0.0)),
                "losses": int(np.sum(delta < 0.0)),
                "draws": int(np.sum(np.isclose(delta, 0.0))),
                "worst_subject_delta": float(delta.min()),
                "late_session_gap": float(
                    (merged["late_session_acc"] - merged["late_session_acc_baseline"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["setting", "coverage", "policy"]).reset_index(
        drop=True
    )


def _load_subject_details(details_dir: Path, subject: int) -> Dict[str, np.ndarray]:
    path = details_dir / f"subject_A{int(subject):02d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing subject detail file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def attach_policy_subject_metrics(
    policy_windows: pd.DataFrame,
    details_dir: str | Path,
) -> pd.DataFrame:
    details_path = Path(details_dir)
    rows: List[Dict] = []
    for (policy, setting, coverage, subject), frame in policy_windows.groupby(
        ["policy", "setting", "coverage", "subject"]
    ):
        details = _load_subject_details(details_path, int(subject))
        y_true = np.asarray(details["y_true"], dtype=np.int64)
        y_pred_ra = np.asarray(details["y_pred_ra"], dtype=np.int64)
        y_pred_w05 = np.asarray(details["y_pred_w05"], dtype=np.int64)
        window_id_by_trial = np.asarray(details["window_id_by_trial"], dtype=np.int64)

        selected_predictions = y_pred_ra.copy()
        selected_action_by_window = {
            int(row.window_id): row.selected_action for row in frame.itertuples(index=False)
        }
        for window_id, action in selected_action_by_window.items():
            if action != "use_partial_alignment":
                continue
            mask = window_id_by_trial == int(window_id)
            selected_predictions[mask] = y_pred_w05[mask]

        late_start = max(0, len(y_true) - max(1, len(y_true) // 3))
        ra_accuracy = float(frame["acc_ra"].mean())
        policy_accuracy = float(np.mean(selected_predictions == y_true))
        rows.append(
            {
                "policy": str(policy),
                "setting": str(setting),
                "coverage": float(coverage),
                "subject": int(subject),
                "policy_accuracy": policy_accuracy,
                "ra_accuracy": ra_accuracy,
                "delta_vs_ra": float(policy_accuracy - ra_accuracy),
                "negative_windows": int(
                    np.sum(frame["selected_accuracy"].to_numpy(dtype=np.float64) < frame["acc_ra"].to_numpy(dtype=np.float64))
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, selected_predictions)
                ),
                "macro_f1": float(f1_score(y_true, selected_predictions, average="macro")),
                "trial_accuracy": policy_accuracy,
                "late_session_acc": float(
                    np.mean(selected_predictions[late_start:] == y_true[late_start:])
                ),
            }
        )
    return pd.DataFrame(rows)
