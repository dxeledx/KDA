# Stage2 Rescue Iteration Summary

## Final Comparison
| method | accuracy_mean | accuracy_std | delta_vs_ra | wins | losses | p_value | effect_size | elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ea | 0.4275 | 0.1332 | +0.0000 | 0 | 0 | nan | nan | 35.77 |
| ra | 0.4275 | 0.1332 | +0.0000 | 0 | 0 | nan | nan | 35.92 |
| fixed/w=1.0 | 0.4275 | 0.1332 | +0.0000 | 0 | 0 | nan | nan | 54.82 |
| adaptive/best | 0.4213 | 0.1287 | -0.0062 | 2 | 5 | 0.2359 | -0.3711 | 59.20 |
| fixed/w=0.5 | 0.4151 | 0.1310 | -0.0123 | 4 | 5 | 0.3008 | -0.4091 | 54.66 |
| noalign | 0.3816 | 0.1136 | -0.0459 | 2 | 7 | 0.0742 | -0.6468 | 17.49 |

## Selected Adaptive Candidate
- Candidate: `adaptive/feedback=sum`
- Stage: `feedback_stability_scan`
- Overrides: `{"conditional": {"bias": 1.0, "ema_smooth_alpha": 0.0, "temperature": 1.0, "weights": [1.2, 0.6, 0.3, 1.2]}, "context": {"cov_eps": 1e-06, "features": ["d_src", "d_tgt", "sigma_recent", "d_geo"]}}`
- Mechanism figure: `results/stage2/rescue/2026-03-06-stage2-rescue/figures/accuracy_vs_weight_best_adaptive.pdf`

## Negative Transfer
- Base adaptive negative-transfer subjects: [1, 3, 4, 5, 6]
- Best adaptive negative-transfer subjects: [1, 3, 4, 6, 7]

## Gates
- gate1: **PASS** — 固定 w=1.0 是否持续压过当前 adaptive，用于判断是否切到保守 gating 主线。
- gate2: **FAIL** — best adaptive 是否已经平均优于 RA 且被试胜场数增加。
- gate3: **FAIL** — 是否满足 3-5% 提升、至少 5/9 被试获益、统计方向支持与开销可解释。

## Fallback Note
- 当前结果尚未满足顶会主线门槛；建议把本轮输出直接转成 failure-analysis / pivot 备忘录输入 stage3。
