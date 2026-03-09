from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.alignment.koopman_alignment import (
    KoopmanShrinkageAffineAligner,
    build_supervised_aligner,
)
from src.evaluation.kcar_analysis import (
    KoopmanOperator,
    compute_transition_residuals,
    fit_koopman_operator,
    fit_subjectwise_global_koopman,
)
from src.features.covariance import compute_covariances
from src.evaluation.ksda_v3 import (
    KSDAV3Fold,
    build_window_slices,
    compute_window_oracle_actions,
    fit_linear_multiclass_selector,
    fit_linear_scalar_proxy,
    load_ksda_v3_folds,
    oracle_usage_stats,
    predict_linear_multiclass_selector,
    predict_linear_scalar_proxy,
)


@dataclass
class TrialSafeAction:
    name: str
    endpoint: str
    alpha: float
    source_mode: str
    target_train_mean: np.ndarray
    target_train_std: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    ema_alpha: float = 0.1
    clip_min: float = 0.67
    clip_max: float = 1.5
    fixed_aligner: object | None = None
    fixed_source_transformed: np.ndarray | None = None
    source_features: np.ndarray | None = None

    def transform_source(self) -> np.ndarray:
        if self.alpha <= 0.0 or self.endpoint == "identity":
            return np.asarray(self.source_features, dtype=np.float64)
        if self.source_mode == "fixed" and self.fixed_aligner is not None:
            transformed = np.asarray(self.fixed_aligner.transform(self.source_features), dtype=np.float64)
            return np.asarray(
                self.source_features + self.alpha * (transformed - self.source_features),
                dtype=np.float64,
            )
        return np.asarray(self.source_features, dtype=np.float64)

    def transform_target_sequence(self, psi_target_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_target_test = np.asarray(psi_target_test, dtype=np.float64)
        transformed_rows = []
        delta_rows = []

        if self.alpha <= 0.0 or self.endpoint == "identity":
            return psi_target_test.copy(), np.zeros(psi_target_test.shape[0], dtype=np.float64)

        if self.endpoint in {"low_rank_affine", "supervised_subspace"}:
            endpoint_features = np.asarray(self.fixed_aligner.transform(psi_target_test), dtype=np.float64)
            transformed = psi_target_test + self.alpha * (endpoint_features - psi_target_test)
            delta = np.linalg.norm(transformed - psi_target_test, axis=1)
            return np.asarray(transformed, dtype=np.float64), np.asarray(delta, dtype=np.float64)

        mean_hist = np.asarray(self.target_train_mean, dtype=np.float64).copy()
        second_hist = mean_hist * mean_hist + np.asarray(self.target_train_std, dtype=np.float64) ** 2
        for psi_t in psi_target_test:
            if self.endpoint == "history_mean_shift":
                endpoint_t = psi_t + 0.5 * (self.source_mean - mean_hist)
            elif self.endpoint == "history_diagonal_scaling":
                var_hist = np.clip(second_hist - mean_hist * mean_hist, 1.0e-8, None)
                std_hist = np.sqrt(var_hist)
                scale = np.clip(self.source_std / std_hist, self.clip_min, self.clip_max)
                endpoint_t = (psi_t - mean_hist) * scale + self.source_mean
            else:
                raise ValueError(f"Unsupported endpoint '{self.endpoint}'")

            transformed_t = psi_t + self.alpha * (endpoint_t - psi_t)
            transformed_rows.append(transformed_t)
            delta_rows.append(float(np.linalg.norm(transformed_t - psi_t)))

            mean_hist = (1.0 - self.ema_alpha) * mean_hist + self.ema_alpha * psi_t
            second_hist = (1.0 - self.ema_alpha) * second_hist + self.ema_alpha * (psi_t * psi_t)

        return np.asarray(transformed_rows, dtype=np.float64), np.asarray(delta_rows, dtype=np.float64)


def _fit_source_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    return features.mean(axis=0), np.clip(features.std(axis=0), 1.0e-8, None)


def build_trial_safe_actions(
    psi_source: np.ndarray,
    psi_target_train: np.ndarray,
    y_source: np.ndarray,
    cov_eps: float = 1.0e-6,
) -> Dict[str, TrialSafeAction]:
    psi_source = np.asarray(psi_source, dtype=np.float64)
    psi_target_train = np.asarray(psi_target_train, dtype=np.float64)
    y_source = np.asarray(y_source, dtype=np.int64)

    source_mean, source_std = _fit_source_statistics(psi_source)
    target_mean, target_std = _fit_source_statistics(psi_target_train)

    fixed_p3 = KoopmanShrinkageAffineAligner(rank=16, shrink=0.3, eps=cov_eps).fit(
        psi_source, psi_target_train
    )
    fixed_p4 = build_supervised_aligner(
        "A1", k=32, reg_lambda=1.0e-4, normalize_output=True
    ).fit(psi_source, y_source)

    actions: Dict[str, TrialSafeAction] = {
        "A0_identity": TrialSafeAction(
            name="A0_identity",
            endpoint="identity",
            alpha=0.0,
            source_mode="raw",
            target_train_mean=target_mean,
            target_train_std=target_std,
            source_mean=source_mean,
            source_std=source_std,
            source_features=psi_source,
        )
    }

    endpoint_specs = [
        ("P1_mean_shift", "history_mean_shift", None),
        ("P2_diag_scaling", "history_diagonal_scaling", None),
        ("P3_low_rank_affine", "low_rank_affine", fixed_p3),
        ("P4_supervised_subspace", "supervised_subspace", fixed_p4),
    ]
    for prefix, endpoint_name, fixed in endpoint_specs:
        for alpha, suffix in [(0.33, "a033"), (0.67, "a067"), (1.0, "a100")]:
            actions[f"{prefix}_{suffix}"] = TrialSafeAction(
                name=f"{prefix}_{suffix}",
                endpoint=endpoint_name,
                alpha=float(alpha),
                source_mode="fixed" if fixed is not None else "raw",
                target_train_mean=target_mean,
                target_train_std=target_std,
                source_mean=source_mean,
                source_std=source_std,
                fixed_aligner=fixed,
                source_features=psi_source,
            )
    return actions


def evaluate_trial_safe_actions(
    psi_source: np.ndarray,
    psi_target_train: np.ndarray,
    psi_target_test: np.ndarray,
    y_source: np.ndarray,
    y_target_test: np.ndarray,
    lda_cls,
    lda_kwargs: Dict[str, object],
    cov_eps: float = 1.0e-6,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    actions = build_trial_safe_actions(psi_source, psi_target_train, y_source, cov_eps=cov_eps)
    results: Dict[str, Dict[str, np.ndarray | float]] = {}
    for name, action in actions.items():
        source_transformed = action.transform_source()
        target_transformed, transform_delta = action.transform_target_sequence(psi_target_test)
        lda = lda_cls(**lda_kwargs).fit(source_transformed, y_source)
        y_pred = lda.predict(target_transformed).astype(np.int64)
        results[name] = {
            "y_pred": y_pred,
            "transform_delta": np.asarray(transform_delta, dtype=np.float64),
            "accuracy": float(np.mean(y_pred == y_target_test)),
            "source_transformed": np.asarray(source_transformed, dtype=np.float64),
            "target_transformed": np.asarray(target_transformed, dtype=np.float64),
        }
    return results


def compute_action_overlap_matrix(predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    names = list(predictions)
    matrix: Dict[str, Dict[str, float]] = {}
    for left in names:
        matrix[left] = {}
        for right in names:
            matrix[left][right] = float(np.mean(predictions[left] == predictions[right]))
    return matrix


def summarize_trial_safe_actions(action_results: Dict[str, Dict[str, np.ndarray | float]], identity_name: str = "A0_identity") -> Dict[str, object]:
    predictions = {name: np.asarray(payload["y_pred"], dtype=np.int64) for name, payload in action_results.items()}
    overlap = compute_action_overlap_matrix(predictions)
    ranked = sorted(action_results.items(), key=lambda item: (float(item[1]["accuracy"]), item[0]), reverse=True)
    top3 = [name.split("_")[0] for name, _ in ranked[:3]]
    non_identity = [name for name in action_results if name != identity_name]
    high_overlap_actions = [
        name for name in non_identity if float(overlap[name][identity_name]) > 0.95
    ]
    low_delta_actions = [
        name
        for name in non_identity
        if float(np.mean(np.asarray(action_results[name]["transform_delta"], dtype=np.float64))) < 1.0e-3
    ]
    same_family_top3 = len(set(top3)) == 1 if len(top3) == 3 else False
    return {
        "best_single_action": ranked[0][0],
        "best_single_accuracy": float(ranked[0][1]["accuracy"]),
        "overlap_vs_identity": {name: float(overlap[name][identity_name]) for name in non_identity},
        "mean_transform_delta": {
            name: float(np.mean(np.asarray(payload["transform_delta"], dtype=np.float64)))
            for name, payload in action_results.items()
        },
        "high_overlap_actions": high_overlap_actions,
        "low_delta_actions": low_delta_actions,
        "same_family_top3": same_family_top3,
        "passed": not (len(high_overlap_actions) >= 2 or len(low_delta_actions) >= 2 or same_family_top3),
    }


def build_causal_teacher_actions(window_oracle_actions: np.ndarray, default_action: int = 0) -> np.ndarray:
    window_oracle_actions = np.asarray(window_oracle_actions, dtype=np.int64)
    teacher = np.empty_like(window_oracle_actions)
    if teacher.size == 0:
        return teacher
    teacher[0] = int(default_action)
    if teacher.size > 1:
        teacher[1:] = window_oracle_actions[:-1]
    return teacher


def trialize_window_actions(window_actions: np.ndarray, total_trials: int, window_size: int) -> np.ndarray:
    window_actions = np.asarray(window_actions, dtype=np.int64)
    expanded = np.zeros(int(total_trials), dtype=np.int64)
    for idx, (start, end) in enumerate(build_window_slices(total_trials, window_size)):
        expanded[start:end] = int(window_actions[idx])
    return expanded


def compute_trial_features(
    z_states: np.ndarray,
    psi_states: np.ndarray,
    source_operator: KoopmanOperator,
    target_operator: KoopmanOperator,
    lda_raw,
    trailing_len: int,
    source_mean_z: np.ndarray,
    source_diag_var_z: np.ndarray,
) -> np.ndarray:
    z_states = np.asarray(z_states, dtype=np.float64)
    psi_states = np.asarray(psi_states, dtype=np.float64)
    rows = []
    running_preds = []
    for trial_idx in range(len(z_states)):
        start = max(0, trial_idx - int(trailing_len) + 1)
        z_hist = z_states[start : trial_idx + 1]
        psi_hist = psi_states[start : trial_idx + 1]
        if len(z_hist) >= 2:
            r_src = float(np.mean(compute_transition_residuals(z_hist, source_operator)))
            r_tgt = float(np.mean(compute_transition_residuals(z_hist, target_operator)))
        else:
            r_src = 0.0
            r_tgt = 0.0
        delta_r = float(r_tgt - r_src)

        prob = np.asarray(lda_raw.predict_proba(psi_hist), dtype=np.float64)
        sorted_prob = np.sort(prob, axis=1)
        if sorted_prob.shape[1] >= 2:
            margin = float(np.mean(sorted_prob[:, -1] - sorted_prob[:, -2]))
        else:
            margin = float(np.mean(sorted_prob[:, -1]))
        pred_hist = np.argmax(prob, axis=1)
        stability = float(np.mean(pred_hist[1:] == pred_hist[:-1])) if len(pred_hist) > 1 else 1.0

        mean_drift = float(np.linalg.norm(z_hist.mean(axis=0) - source_mean_z))
        diag_var = np.var(z_hist, axis=0) if len(z_hist) > 1 else np.zeros_like(source_diag_var_z)
        cov_drift = float(np.linalg.norm(diag_var - source_diag_var_z))
        drift = float(mean_drift + 0.1 * cov_drift)
        rows.append(np.asarray([r_src, r_tgt, delta_r, margin, drift, stability], dtype=np.float64))
        running_preds.append(pred_hist[-1])
    return np.asarray(rows, dtype=np.float64)


def load_trial_safe_fold_state(
    fold: KSDAV3Fold,
    projector,
    lda_cls,
    lda_kwargs: Dict[str, object],
    cov_eps: float = 1.0e-6,
) -> Dict[str, object]:
    psi_source = projector.transform(fold.cov_source)
    psi_target_train = projector.transform(fold.cov_target_train)
    psi_target_test = projector.transform(fold.cov_target_test)
    z_source = projector.transform_tangent(fold.cov_source)
    z_target_train = projector.transform_tangent(fold.cov_target_train)
    z_target_test = projector.transform_tangent(fold.cov_target_test)

    action_results = evaluate_trial_safe_actions(
        psi_source,
        psi_target_train,
        psi_target_test,
        fold.y_source,
        fold.y_target_test,
        lda_cls,
        lda_kwargs,
        cov_eps=cov_eps,
    )

    source_blocks = []
    start = 0
    for length in fold.source_block_lengths:
        source_blocks.append(z_source[start : start + int(length)])
        start += int(length)
    source_operator = fit_subjectwise_global_koopman(source_blocks, ridge_alpha=1.0e-3)
    target_operator = fit_koopman_operator(z_target_train, ridge_alpha=1.0e-3)
    source_mean_z = np.mean(np.concatenate(source_blocks, axis=0), axis=0)
    source_diag_var_z = np.var(np.concatenate(source_blocks, axis=0), axis=0)
    raw_lda = lda_cls(**lda_kwargs).fit(psi_source, fold.y_source)

    return {
        "psi_source": psi_source,
        "psi_target_train": psi_target_train,
        "psi_target_test": psi_target_test,
        "z_target_test": z_target_test,
        "action_results": action_results,
        "action_names": list(action_results.keys()),
        "summary": summarize_trial_safe_actions(action_results),
        "source_operator": source_operator,
        "target_operator": target_operator,
        "source_mean_z": source_mean_z,
        "source_diag_var_z": source_diag_var_z,
        "raw_lda": raw_lda,
    }


def load_custom_ksda_fold(loader, source_subjects: Sequence[int], target_subject: int, pre, cov_eps: float) -> KSDAV3Fold:
    X_source_blocks, y_source_blocks = [], []
    for subject in source_subjects:
        X_train_subject, y_train_subject = loader.load_subject(int(subject), split="train")
        X_source_blocks.append(X_train_subject)
        y_source_blocks.append(y_train_subject)

    X_source_raw = np.concatenate(X_source_blocks, axis=0)
    y_source = np.concatenate(y_source_blocks, axis=0)
    X_target_train_raw, _ = loader.load_subject(int(target_subject), split="train")
    X_target_test_raw, y_target_test = loader.load_subject(int(target_subject), split="test")

    X_source = pre.fit(X_source_raw, y_source).transform(X_source_raw)
    X_target_train = pre.transform(X_target_train_raw)
    X_target_test = pre.transform(X_target_test_raw)

    return KSDAV3Fold(
        target_subject=int(target_subject),
        cov_source=compute_covariances(X_source, eps=cov_eps),
        y_source=np.asarray(y_source, dtype=np.int64),
        cov_target_train=compute_covariances(X_target_train, eps=cov_eps),
        cov_target_test=compute_covariances(X_target_test, eps=cov_eps),
        y_target_test=np.asarray(y_target_test, dtype=np.int64),
        source_block_lengths=[block.shape[0] for block in X_source_blocks],
    )


def compute_window_oracle_for_actions(
    y_true: np.ndarray,
    action_predictions: Dict[str, np.ndarray],
    window_size: int,
) -> Dict[str, object]:
    action_names = list(action_predictions.keys())
    score_rows = []
    oracle_indices = []
    oracle_pred = np.zeros_like(y_true, dtype=np.int64)
    for window_idx, (start, end) in enumerate(build_window_slices(len(y_true), window_size)):
        y_window = y_true[start:end]
        scores = np.asarray(
            [float(np.mean(action_predictions[name][start:end] == y_window)) for name in action_names],
            dtype=np.float64,
        )
        chosen_idx, oracle_acc = compute_window_oracle_actions(scores[None, :])
        chosen_idx = int(chosen_idx[0])
        oracle_indices.append(chosen_idx)
        oracle_pred[start:end] = action_predictions[action_names[chosen_idx]][start:end]
        score_rows.append(
            {
                "window_id": int(window_idx),
                "oracle_action_idx": chosen_idx,
                "oracle_action_name": action_names[chosen_idx],
                "oracle_acc": float(oracle_acc),
                **{f"acc_{name}": float(score) for name, score in zip(action_names, scores)},
            }
        )
    return {
        "window_rows": score_rows,
        "oracle_action_indices": np.asarray(oracle_indices, dtype=np.int64),
        "oracle_pred": oracle_pred,
        "action_names": action_names,
    }


def teacher_agreement(current_window_actions: np.ndarray, teacher_window_actions: np.ndarray) -> float:
    current_window_actions = np.asarray(current_window_actions, dtype=np.int64)
    teacher_window_actions = np.asarray(teacher_window_actions, dtype=np.int64)
    if current_window_actions.size == 0:
        return 0.0
    return float(np.mean(current_window_actions == teacher_window_actions))


def build_causal_trialized_teacher_from_state(
    fold: KSDAV3Fold,
    state: Dict[str, object],
    window_size: int,
    default_action_name: str = "A0_identity",
) -> Dict[str, object]:
    action_predictions = {
        name: np.asarray(payload["y_pred"], dtype=np.int64)
        for name, payload in state["action_results"].items()
    }
    action_names = list(action_predictions.keys())
    action_to_idx = {name: idx for idx, name in enumerate(action_names)}
    window_oracle = compute_window_oracle_for_actions(
        np.asarray(fold.y_target_test, dtype=np.int64),
        action_predictions,
        window_size=window_size,
    )
    oracle_action_indices = np.asarray(window_oracle["oracle_action_indices"], dtype=np.int64)
    teacher_indices = build_causal_teacher_actions(
        oracle_action_indices,
        default_action=action_to_idx[default_action_name],
    )
    teacher_trial_indices = trialize_window_actions(teacher_indices, len(fold.y_target_test), window_size)
    window_oracle_trial_indices = trialize_window_actions(oracle_action_indices, len(fold.y_target_test), window_size)

    teacher_pred = np.zeros_like(fold.y_target_test, dtype=np.int64)
    oracle_pred = np.zeros_like(fold.y_target_test, dtype=np.int64)
    for action_name, action_idx in action_to_idx.items():
        teacher_mask = teacher_trial_indices == action_idx
        oracle_mask = window_oracle_trial_indices == action_idx
        teacher_pred[teacher_mask] = action_predictions[action_name][teacher_mask]
        oracle_pred[oracle_mask] = action_predictions[action_name][oracle_mask]

    return {
        "action_names": action_names,
        "action_to_idx": action_to_idx,
        "window_oracle_action_indices": oracle_action_indices,
        "teacher_action_indices": teacher_indices,
        "teacher_trial_indices": teacher_trial_indices,
        "window_oracle_trial_indices": window_oracle_trial_indices,
        "teacher_pred": teacher_pred,
        "window_oracle_pred": oracle_pred,
        "window_rows": window_oracle["window_rows"],
        "teacher_vs_window_agreement": teacher_agreement(window_oracle_trial_indices, teacher_trial_indices),
    }


def paired_wins(left: Iterable[float], right: Iterable[float]) -> Dict[str, int | float]:
    left = np.asarray(list(left), dtype=np.float64)
    right = np.asarray(list(right), dtype=np.float64)
    delta = left - right
    return {
        "mean_delta": float(delta.mean()) if delta.size else 0.0,
        "wins": int(np.sum(delta > 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "draws": int(np.sum(np.isclose(delta, 0.0))),
    }


def summarize_rank_scan_metrics(
    *,
    best_single_action: str,
    best_single_accuracy: float,
    overlap_vs_identity: Dict[str, float],
) -> Dict[str, object]:
    overlaps = {name: float(value) for name, value in overlap_vs_identity.items()}
    high_overlap = [name for name, value in overlaps.items() if value > 0.95]
    max_overlap = max(overlaps.values()) if overlaps else 0.0
    mean_overlap = float(np.mean(list(overlaps.values()))) if overlaps else 0.0
    return {
        "best_single_action": str(best_single_action),
        "best_single_accuracy": float(best_single_accuracy),
        "num_high_overlap_actions": int(len(high_overlap)),
        "high_overlap_actions": high_overlap,
        "max_overlap_vs_identity": float(max_overlap),
        "mean_overlap_vs_identity": float(mean_overlap),
    }


def resolve_representation_config(
    base_representation: Dict[str, object],
    pca_rank: int | None = None,
    lifting: str | None = None,
) -> Dict[str, object]:
    return {
        "pca_rank": int(pca_rank if pca_rank is not None else base_representation["pca_rank"]),
        "lifting": str(lifting if lifting is not None else base_representation["lifting"]),
    }
