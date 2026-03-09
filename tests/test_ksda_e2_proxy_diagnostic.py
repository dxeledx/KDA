import json
from pathlib import Path

import pandas as pd


def test_e2_proxy_diag_parser_accepts_roots():
    from experiments import ksda_exp_e2_proxy_diagnostic

    parser = ksda_exp_e2_proxy_diagnostic.build_argparser()
    args = parser.parse_args(
        [
            "--run-name",
            "demo",
            "--e0-root",
            "results/e0/demo",
            "--refresh-root",
            "results/e2_diag/demo",
        ]
    )
    assert args.run_name == "demo"
    assert args.e0_root == "results/e0/demo"
    assert args.refresh_root == "results/e2_diag/demo"


def test_summarize_proxy_scores_prefers_better_proxy():
    from experiments.ksda_exp_e2_proxy_diagnostic import summarize_proxy_scores

    pairwise_df = pd.DataFrame(
        [
            {"source": 1, "target": 3, "accuracy": 0.8, "ra_accuracy": 0.75, "good": 0.9, "bad": 0.1},
            {"source": 2, "target": 3, "accuracy": 0.4, "ra_accuracy": 0.35, "good": 0.3, "bad": 0.8},
            {"source": 1, "target": 4, "accuracy": 0.7, "ra_accuracy": 0.65, "good": 0.8, "bad": 0.2},
            {"source": 2, "target": 4, "accuracy": 0.2, "ra_accuracy": 0.25, "good": 0.1, "bad": 0.7},
        ]
    )

    proxy_summary, target_summary = summarize_proxy_scores(
        pairwise_df=pairwise_df,
        proxy_columns=["good", "bad"],
    )

    good = proxy_summary.loc[proxy_summary["proxy"] == "good"].iloc[0]
    bad = proxy_summary.loc[proxy_summary["proxy"] == "bad"].iloc[0]
    assert float(good["rbid_vs_behavior"]) < float(bad["rbid_vs_behavior"])
    assert float(good["mean_target_corr_behavior_spearman"]) > float(bad["mean_target_corr_behavior_spearman"])
    assert set(target_summary["proxy"]) == {"good", "bad"}


def test_build_summary_and_write_outputs_for_proxy_diag(tmp_path: Path):
    from experiments.ksda_exp_e2_proxy_diagnostic import build_summary, write_outputs

    proxy_summary = pd.DataFrame(
        [
            {
                "proxy": "proxy_train_mean_cosine",
                "rbid_vs_behavior": 0.30,
                "tail_rbid_vs_behavior": 0.55,
                "pearson_vs_behavior": 0.25,
                "spearman_vs_behavior": 0.30,
                "mean_target_corr_behavior_spearman": 0.40,
                "rbid_vs_ra": 0.28,
                "mean_target_corr_ra_spearman": 0.45,
            },
            {
                "proxy": "proxy_test_cka",
                "rbid_vs_behavior": 0.26,
                "tail_rbid_vs_behavior": 0.50,
                "pearson_vs_behavior": 0.35,
                "spearman_vs_behavior": 0.38,
                "mean_target_corr_behavior_spearman": 0.52,
                "rbid_vs_ra": 0.29,
                "mean_target_corr_ra_spearman": 0.40,
            },
        ]
    )
    summary = build_summary(run_dir=tmp_path, proxy_summary=proxy_summary)
    pairwise_df = pd.DataFrame([{"source": 1, "target": 2, "accuracy": 0.5}])
    target_df = pd.DataFrame([{"target": 2, "proxy": "proxy_test_cka", "corr_behavior_spearman": 1.0}])

    write_outputs(
        output_dir=tmp_path,
        summary=summary,
        pairwise_df=pairwise_df,
        proxy_summary=proxy_summary,
        target_summary=target_df,
        memo_text="demo proxy diag",
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "pairwise_proxy_scores.csv").exists()
    assert (tmp_path / "proxy_summary.csv").exists()
    assert (tmp_path / "target_proxy_diagnostics.csv").exists()
    assert (tmp_path / "memo.md").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["best_proxy_by_behavior"]["proxy"] == "proxy_test_cka"
    assert saved["current_training_proxy"]["proxy"] == "proxy_train_mean_cosine"
