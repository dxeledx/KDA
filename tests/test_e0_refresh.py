import json
from pathlib import Path

import pandas as pd


def test_baseline_scripts_accept_custom_output_args():
    from experiments import baseline_csp_lda, baseline_ea, baseline_ra

    for module in (baseline_csp_lda, baseline_ea, baseline_ra):
        parser = module.build_argparser()
        args = parser.parse_args(["--out-dir", "tmp/out", "--fig-dir", "tmp/figs"])
        assert args.out_dir == "tmp/out"
        assert args.fig_dir == "tmp/figs"


def test_phenomenon_verification_parser_supports_custom_roots_and_rank():
    from experiments import phenomenon_verification

    parser = phenomenon_verification.build_argparser()
    args = parser.parse_args(
        [
            "--classical-root",
            "tmp/classical",
            "--output-root",
            "tmp/phenomenon/r48",
            "--koopman-pca-rank",
            "48",
        ]
    )
    assert args.classical_root == "tmp/classical"
    assert args.output_root == "tmp/phenomenon/r48"
    assert args.koopman_pca_rank == 48


def test_build_e0_main_table_joins_classical_and_koopman_metrics(tmp_path: Path):
    from experiments.e0_refresh_baselines import build_e0_main_table

    classical_root = tmp_path / "baselines"
    for method, acc, kappa in (
        ("noalign", 0.38, 0.17),
        ("ea", 0.42, 0.23),
        ("ra", 0.43, 0.24),
    ):
        method_dir = classical_root / method
        method_dir.mkdir(parents=True)
        (method_dir / "summary.json").write_text(
            json.dumps(
                {
                    "accuracy": {"mean": acc, "std": 0.10},
                    "kappa": {"mean": kappa, "std": 0.11},
                }
            ),
            encoding="utf-8",
        )

    phenomenon_root = tmp_path / "phenomenon_r48"
    phenomenon_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "Koopman-noalign",
                "accuracy_mean": 0.46,
                "accuracy_std": 0.12,
                "kappa_mean": 0.28,
                "kappa_std": 0.10,
            },
            {
                "method": "Static Koopman aligner",
                "accuracy_mean": 0.47,
                "accuracy_std": 0.11,
                "kappa_mean": 0.29,
                "kappa_std": 0.09,
            },
        ]
    ).to_csv(phenomenon_root / "loso_method_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "No Alignment",
                "method_key": "noalign",
                "pearson_r": 0.39,
                "rbid": 0.31,
                "rbid_pos": 0.15,
                "rbid_neg": 0.16,
                "tail_rbid": 0.62,
            },
            {
                "method": "EA",
                "method_key": "ea",
                "pearson_r": 0.52,
                "rbid": 0.29,
                "rbid_pos": 0.14,
                "rbid_neg": 0.15,
                "tail_rbid": 0.57,
            },
            {
                "method": "RA",
                "method_key": "ra",
                "pearson_r": 0.52,
                "rbid": 0.30,
                "rbid_pos": 0.15,
                "rbid_neg": 0.15,
                "tail_rbid": 0.57,
            },
            {
                "method": "Koopman-noalign",
                "method_key": "koopman-noalign",
                "pearson_r": 0.09,
                "rbid": 0.35,
                "rbid_pos": 0.17,
                "rbid_neg": 0.18,
                "tail_rbid": 0.68,
            },
            {
                "method": "Static Koopman aligner",
                "method_key": "static-koopman-aligner",
                "pearson_r": 0.27,
                "rbid": 0.34,
                "rbid_pos": 0.17,
                "rbid_neg": 0.17,
                "tail_rbid": 0.66,
            },
        ]
    ).to_csv(phenomenon_root / "rbid_method_comparison.csv", index=False)

    main_table = build_e0_main_table(classical_root, phenomenon_root, koopman_rank=48)

    assert list(main_table["method"]) == [
        "No Alignment",
        "EA",
        "RA",
        "Koopman-noalign-r48",
        "Static Koopman aligner-r48",
    ]
    assert list(main_table.columns) == [
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
    assert main_table.isna().sum().sum() == 0


def test_build_historical_koopman_table_compares_r16_and_r48(tmp_path: Path):
    from experiments.e0_refresh_baselines import build_historical_koopman_table

    r16_root = tmp_path / "r16"
    r48_root = tmp_path / "r48"
    r16_root.mkdir()
    r48_root.mkdir()

    pd.DataFrame(
        [
            {"method": "Koopman-noalign", "accuracy_mean": 0.41, "accuracy_std": 0.14, "kappa_mean": 0.21, "kappa_std": 0.10},
            {"method": "Static Koopman aligner", "accuracy_mean": 0.42, "accuracy_std": 0.13, "kappa_mean": 0.22, "kappa_std": 0.11},
        ]
    ).to_csv(r16_root / "loso_method_summary.csv", index=False)
    pd.DataFrame(
        [
            {"method": "Koopman-noalign", "pearson_r": 0.08, "rbid": 0.35, "tail_rbid": 0.68},
            {"method": "Static Koopman aligner", "pearson_r": 0.27, "rbid": 0.34, "tail_rbid": 0.66},
        ]
    ).to_csv(r16_root / "rbid_method_comparison.csv", index=False)

    pd.DataFrame(
        [
            {"method": "Koopman-noalign", "accuracy_mean": 0.45, "accuracy_std": 0.12, "kappa_mean": 0.26, "kappa_std": 0.09},
            {"method": "Static Koopman aligner", "accuracy_mean": 0.47, "accuracy_std": 0.11, "kappa_mean": 0.29, "kappa_std": 0.08},
        ]
    ).to_csv(r48_root / "loso_method_summary.csv", index=False)
    pd.DataFrame(
        [
            {"method": "Koopman-noalign", "pearson_r": 0.11, "rbid": 0.33, "tail_rbid": 0.63},
            {"method": "Static Koopman aligner", "pearson_r": 0.30, "rbid": 0.31, "tail_rbid": 0.59},
        ]
    ).to_csv(r48_root / "rbid_method_comparison.csv", index=False)

    hist = build_historical_koopman_table(r16_root, r48_root)

    assert list(hist["method"]) == ["Koopman-noalign", "Static Koopman aligner"]
    assert "accuracy_mean_r16" in hist.columns
    assert "accuracy_mean_r48" in hist.columns
    assert hist.loc[0, "rbid_r48"] < hist.loc[0, "rbid_r16"]


def test_render_e0_memo_does_not_require_tabulate():
    from experiments.e0_refresh_baselines import render_e0_memo

    main = pd.DataFrame(
        [
            {
                "method": "No Alignment",
                "accuracy_mean": 0.38,
                "accuracy_std": 0.10,
                "kappa_mean": 0.17,
                "kappa_std": 0.11,
                "rbid": 0.31,
                "rbid_pos": 0.15,
                "rbid_neg": 0.16,
                "tail_rbid": 0.62,
                "pearson_r": 0.39,
            },
            {
                "method": "Static Koopman aligner-r48",
                "accuracy_mean": 0.47,
                "accuracy_std": 0.11,
                "kappa_mean": 0.29,
                "kappa_std": 0.09,
                "rbid": 0.31,
                "rbid_pos": 0.15,
                "rbid_neg": 0.16,
                "tail_rbid": 0.59,
                "pearson_r": 0.48,
            },
        ]
    )
    hist = pd.DataFrame(
        [
            {
                "method": "Static Koopman aligner",
                "accuracy_mean_r16": 0.41,
                "accuracy_mean_r48": 0.47,
                "rbid_r16": 0.34,
                "rbid_r48": 0.31,
            }
        ]
    )

    memo = render_e0_memo(Path("results/e0/demo"), main, hist)
    assert "E0 Baseline Refresh Memo" in memo
    assert "Static Koopman aligner-r48" in memo
    assert "E1 control method is fixed" in memo
