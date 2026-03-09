# E2 Diagnostic Baseline Refresh Memo

**Run dir**: `results/e2_diag/2026-03-09-e2diag-r1`  
**Pairwise protocol**: `target-global pooled-source per target`  
**Original E1 pairwise protocol**: `per-source per-target refit`

## Key deltas

- Refresh vs original E1 RBID: `-0.0119`
- Refresh vs original E1 Tail-RBID: `0.0688`
- E2 vs refresh RBID: `0.0159`
- E2 vs refresh Tail-RBID: `0.0043`
- E2 vs refresh pairwise accuracy mean: `-0.0060`
- E2 vs refresh Pearson-r: `0.0117`

## Mean target-wise rank diagnostics

method,corr_ra_vs_accuracy_spearman,corr_ra_vs_cka_spearman,corr_accuracy_vs_cka_spearman
Conservative Koopman aligner-r48 (original pairwise),0.5906942407400534,0.33692524654062894,0.3122452801160519
Conservative Koopman aligner-r48 (target-global refresh),0.5377159319231436,0.4019768668359023,0.3044234728044594
RBID-aware Conservative Koopman aligner-r48,0.5012220925325502,0.39655519976396597,0.25669320185194633

## Verdict

- E2 beats refreshed baseline on RBID: `False`
- E2 beats refreshed baseline on Tail-RBID: `False`
- E2 beats refreshed baseline on pairwise accuracy: `False`
