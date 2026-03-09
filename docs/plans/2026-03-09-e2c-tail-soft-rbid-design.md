# E2c Tail-Aware Soft-RBID Design

**Date**: 2026-03-09  
**Scope**: `E2c` 只在 `E2b` 的 `Soft-RBID + Huber` 上加入 tail weighting；不改 prior，不改 proxy，不改协议。

## Context

- `E2a` 说明只换 proxy 不够。
- `E2b` 说明只换 surrogate 形式也不够，但 `Tail-RBID` 相对 `E2a` 有轻微回落。
- 因此 `E2c` 的唯一合理增量是：直接把 tail 控制写进 mismatch surrogate。

## Goal

保持以下要素不变：

- behavior prior：仍使用 `E0/RA` pairwise accuracy；
- representation proxy：仍使用 `mean+dyn` score；
- aligner：仍使用 `KoopmanConservativeResidualAligner`；
- protocol：仍使用 `target-global pooled-source per target`；
- surrogate 主体：仍使用 `Soft-RBID + Huber`。

唯一新增：**tail-aware weighting**。

## E2c Definition

### 1. Behavior rank target

对固定 target `j`，先把 `beh(i,j)` 做成归一化 rank target `r_beh`，与 `RBID` 的 per-target rank normalization 一致。

### 2. Soft representation rank

继续用 `E2b` 的 soft rank：

`r_rep_i = soft_rank_tau(u_ij(A))`

### 3. Tail weighting

对固定 target `j`，令：

- `q_j = quantile(r_beh, alpha)`
- `w_i = 1 + lambda_tail * 1[r_beh_i <= q_j]`

解释：

- `r_beh` 低的 source，表示行为上本来就更难迁移；
- 这些 pair 更容易成为 mismatch 的危险尾部；
- 因此在 surrogate 里对它们施加更高权重。

### 4. Tail-Aware Soft-RBID

`L_tail-soft-rbid = sum_i w_i * huber(r_rep_i - r_beh_i; delta) / sum_i w_i`

其中：

- `alpha = rank_tail_quantile`
- `lambda_tail = rank_tail_weight`

## Defaults

- `rank_loss_mode = "tail_soft_rbid_huber"`
- `rank_tail_quantile = 0.25`
- `rank_tail_weight = 2.0`

## Non-Goals

- 不改 `RA prior`
- 不改 `mean+dyn` proxy
- 不改 `tau / delta` 的 sweep 策略
- 不引入 listwise prior blending
- 不进入 `E3 / E4`

## Decision Value

`E2c` 的实验价值很直接：

- 若 `Tail-RBID` 明显下降，说明当前缺的不是更复杂 proxy，而是 tail control；
- 若仍然不行，说明当前瓶颈不在“是否强调 tail”，而在 prior/proxy 本体。
