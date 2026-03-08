import pandas as pd
import pytest

from src.evaluation.stage2_analysis import summarize_against_reference


def test_summarize_against_reference_tracks_wins_and_delta():
    candidate = pd.DataFrame(
        {
            "target_subject": [1, 2, 3],
            "accuracy": [0.70, 0.40, 0.60],
        }
    )
    reference = pd.DataFrame(
        {
            "target_subject": [1, 2, 3],
            "accuracy": [0.50, 0.45, 0.55],
        }
    )

    summary = summarize_against_reference(candidate, reference, metric="accuracy")

    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["draws"] == 0
    assert summary["mean"] == pytest.approx(candidate["accuracy"].mean())
    assert summary["delta_vs_reference"] == pytest.approx(candidate["accuracy"].mean() - reference["accuracy"].mean())
