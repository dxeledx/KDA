import json
from pathlib import Path

import pandas as pd


def test_e2_diag_parser_accepts_all_roots():
    from experiments import ksda_exp_e2_diagnostic_refresh

    parser = ksda_exp_e2_diagnostic_refresh.build_argparser()
    args = parser.parse_args(
        [
            "--run-name",
            "demo",
            "--e0-root",
            "results/e0/demo",
            "--e1-root",
            "results/e1/demo",
            "--e2-root",
            "results/e2/demo",
        ]
    )
    assert args.run_name == "demo"
    assert args.e0_root == "results/e0/demo"
    assert args.e1_root == "results/e1/demo"
    assert args.e2_root == "results/e2/demo"


def test_compute_target_rank_diagnostics_emits_expected_methods():
    from experiments.ksda_exp_e2_diagnostic_refresh import compute_target_rank_diagnostics

    ra_pairwise = pd.DataFrame(
        [
            {"source": 1, "target": 3, "accuracy": 0.8},
            {"source": 2, "target": 3, "accuracy": 0.4},
            {"source": 1, "target": 4, "accuracy": 0.6},
            {"source": 2, "target": 4, "accuracy": 0.2},
        ]
    )
    original = pd.DataFrame(
        [
            {"source": 1, "target": 3, "accuracy": 0.7, "cka": 0.5, "method": "orig"},
            {"source": 2, "target": 3, "accuracy": 0.3, "cka": 0.2, "method": "orig"},
            {"source": 1, "target": 4, "accuracy": 0.4, "cka": 0.4, "method": "orig"},
            {"source": 2, "target": 4, "accuracy": 0.5, "cka": 0.6, "method": "orig"},
        ]
    )
    refreshed = pd.DataFrame(
        [
            {"source": 1, "target": 3, "accuracy": 0.75, "cka": 0.55, "method": "refresh"},
            {"source": 2, "target": 3, "accuracy": 0.35, "cka": 0.25, "method": "refresh"},
            {"source": 1, "target": 4, "accuracy": 0.55, "cka": 0.45, "method": "refresh"},
            {"source": 2, "target": 4, "accuracy": 0.25, "cka": 0.2, "method": "refresh"},
        ]
    )
    e2 = pd.DataFrame(
        [
            {"source": 1, "target": 3, "accuracy": 0.74, "cka": 0.58, "method": "e2"},
            {"source": 2, "target": 3, "accuracy": 0.2, "cka": 0.4, "method": "e2"},
            {"source": 1, "target": 4, "accuracy": 0.52, "cka": 0.7, "method": "e2"},
            {"source": 2, "target": 4, "accuracy": 0.22, "cka": 0.1, "method": "e2"},
        ]
    )

    diagnostics = compute_target_rank_diagnostics(
        ra_pairwise=ra_pairwise,
        method_pairwise={
            "orig": original,
            "refresh": refreshed,
            "e2": e2,
        },
    )

    assert set(diagnostics["method"]) == {"orig", "refresh", "e2"}
    assert {
        "target",
        "method",
        "corr_ra_vs_accuracy_spearman",
        "corr_accuracy_vs_cka_spearman",
    }.issubset(diagnostics.columns)


def test_build_summary_and_write_outputs_for_e2_diag(tmp_path: Path):
    from experiments.ksda_exp_e2_diagnostic_refresh import build_summary, write_outputs

    summary = build_summary(
        run_dir=tmp_path,
        e1_original_row={
            "method": "Conservative Koopman aligner-r48",
            "rbid": 0.3056,
            "tail_rbid": 0.5503,
            "pearson_r": 0.2506,
            "pairwise_accuracy_mean": 0.34,
        },
        refreshed_row={
            "method": "Conservative Koopman aligner-r48 (target-global refresh)",
            "rbid": 0.3111,
            "tail_rbid": 0.6000,
            "pearson_r": 0.3200,
            "pairwise_accuracy_mean": 0.33,
        },
        e2_row={
            "method": "RBID-aware Conservative Koopman aligner-r48",
            "rbid": 0.3095,
            "tail_rbid": 0.6234,
            "pearson_r": 0.3696,
            "pairwise_accuracy_mean": 0.317,
        },
    )
    pairwise_panel = pd.DataFrame([{"source": 1, "target": 2, "accuracy": 0.5, "cka": 0.1, "method": "x"}])
    rbid_panel = pd.DataFrame([{"method": "x", "rbid": 0.3, "tail_rbid": 0.6, "pearson_r": 0.2}])
    diagnostics = pd.DataFrame(
        [{"target": 2, "method": "x", "corr_ra_vs_accuracy_spearman": 1.0, "corr_accuracy_vs_cka_spearman": 1.0}]
    )

    write_outputs(
        output_dir=tmp_path,
        summary=summary,
        pairwise_panel=pairwise_panel,
        rbid_panel=rbid_panel,
        diagnostics=diagnostics,
        memo_text="demo diag",
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "pairwise_scores.csv").exists()
    assert (tmp_path / "rbid_method_comparison.csv").exists()
    assert (tmp_path / "target_rank_diagnostics.csv").exists()
    assert (tmp_path / "memo.md").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert "delta_e2_vs_refresh" in saved
    assert saved["protocol_match"]["pairwise_protocol"] == "target-global pooled-source per target"
