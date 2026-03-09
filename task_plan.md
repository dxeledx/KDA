# Task Plan: KSDA mismatch-first experiment map

## Goal
把当前研究主线、实验边界、下一阶段实验顺序和记录模板固定下来，后续所有实验按同一张 map 推进，避免偏题或遗忘。

## Phases
- [x] Phase 1: 同步当前仓库进展
- [x] Phase 2: 固定主问题与边界
- [x] Phase 3: 完成 E0 baseline refresh
- [x] Phase 4: 完成 E1 conservative aligner
- [ ] Phase 5: 决定是否进入算子修正

## Key Questions
1. 当前主实验到底要证明“什么方法有效”，而不是什么都试一点？
2. 哪些实验属于论文主线，哪些只适合附录或后续工作？
3. 进入下一阶段前，哪些 gate 必须先通过？

## Decisions Made
- 采用两层结构：`experiment map` 负责方向与 gate，`experiment ledger` 负责逐轮实验记录。
- 当前论文主线先做保守的、静态或窗口级的 `RBID-aware Koopman aligner`，不回到 trial-level controller。
- 当前主指标锁定为 `RBID / Tail-RBID / LOSO accuracy`；`K-RBID` 暂时只作为局部诊断量。

## Errors Encountered
- `results/ksda/exp_d1p5/2026-03-09-ksda-d1p5-r48-r3/summary.json` 与若干 CSV 为空；当前以 `docs/KSDA/21-exp-d1t-r48-d1p5-results-memo.md` 的人工结论为准，后续若重跑需补齐机器可读摘要。

## Status
**Currently in Phase 5** - 已完成 `E2c`。结论是：只替换 proxy（`E2a`）、只替换 surrogate（`E2b`）、以及只加 tail weighting（`E2c`）都不足以救回 `E2`。当前更合理的下一步不是进入 `E3 / E4`，而是回到 `behavior prior + representation proxy` 组合本身重诊断。
