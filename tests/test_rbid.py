import numpy as np


def test_rbid_decomposes_into_positive_and_negative_parts():
    from src.evaluation.rbid import compute_rbid_from_pairwise

    rep = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.8, 1.0, 0.2],
            [0.3, 0.4, 1.0],
        ],
        dtype=np.float64,
    )
    beh = np.array(
        [
            [1.0, 0.2, 0.8],
            [0.1, 1.0, 0.7],
            [0.6, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    out = compute_rbid_from_pairwise(rep, beh)
    assert np.isclose(out["rbid"], out["rbid_pos"] + out["rbid_neg"])
    assert out["tail_rbid"] >= out["rbid"]


def test_local_k_rbid_is_zero_when_rankings_match():
    from src.evaluation.rbid import summarize_local_k_rbid

    geometry = np.array([[0.1, 0.5, 0.2], [0.8, 0.2, 0.1]], dtype=np.float64)
    behavior = np.array([[0.2, 0.9, 0.4], [0.7, 0.3, 0.2]], dtype=np.float64)
    out = summarize_local_k_rbid(geometry, behavior)
    assert np.allclose(out["k_rbid_per_window"], 0.0)
