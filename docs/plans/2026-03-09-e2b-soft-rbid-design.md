# E2b Soft-RBID Design

**Date**: 2026-03-09  
**Scope**: `E2b` 只改变 mismatch surrogate；不改 prior，不改 aligner 参数化，不改评估协议。

## Context

- 当前分支里的 `E2` 已经使用了 target-wise `pairwise logistic ranking loss`。
- `E2a` 已验证：只把 proxy 从 `aligned mean cosine` 换成 `mean+dyn` 结构化分数，不足以改善 `RBID / Tail-RBID`。
- 因此 `E2b` 必须真正改 `surrogate form`，而不是再次重复 pairwise logistic。

## Goal

在保持以下要素不变的前提下，把 mismatch loss 从 pairwise ordering surrogate 改成更接近 exact `RBID` 的 `Soft-RBID`：

- behavior prior：仍使用 `E0/RA` pairwise accuracy；
- representation proxy：仍使用 `E2a` 的 `mean+dyn` score；
- aligner：仍使用 `KoopmanConservativeResidualAligner`；
- protocol：仍使用 `target-global pooled-source per target`。

## Recommended E2b Definition

### 1. Representation score

对固定 target `j`，对每个 source `i` 继续使用 `E2a` 的结构化分数：

`u_ij(A) = - alpha * mean_sq_dist_ij(A) - beta * dyn_resid_ij(A)`

其中 `alpha=rank_mean_weight`，`beta=rank_dyn_weight`。

### 2. Behavior rank target

对同一 target `j`，把 `RA` 的 raw pairwise accuracy `beh(i,j)` 变成归一化 rank target：

- source 越能迁移到 target，rank 越高；
- rank 范围归一化到 `[0, 1]`；
- 与 `RBID` 评估中的 per-target rank normalization 保持一致。

### 3. Soft representation rank

把 `u_j = [u_1j, ..., u_mj]` 映射成 soft rank：

`r_rep_i = (sum_k sigmoid((u_i - u_k) / tau) - 0.5) / (m - 1)`

性质：

- 最大分数接近 `1`，最小分数接近 `0`；
- `tau` 越小越接近硬排序；
- 对当前 `numpy + 手写梯度` 代码栈可实现。

### 4. Soft-RBID loss

对固定 target `j`，令 `r_beh` 为 behavior rank target，`r_rep` 为 soft representation rank，定义：

`L_soft-rbid = mean_i huber(r_rep_i - r_beh_i; delta)`

说明：

- 这是 exact `RBID = mean |rank_rep - rank_beh|` 的平滑替代；
- 比当前 pairwise logistic 更直接对应论文指标；
- `delta` 用于稳定优化，避免离群梯度。

## Non-Goals

- 不引入 tail weighting；那是 `E2c`。
- 不混入 `E1` self-prior。
- 不改 proxy 为 covariance-aware score。
- 不改 LOSO / pairwise protocol。

## Expected Decision Value

`E2b` 的意义是把“proxy 不够好”与“surrogate 形式不够像 RBID”拆开：

- 若 `E2b > E2a`，说明 surrogate form 确实重要；
- 若 `E2b ≈ E2a`，说明当前瓶颈仍主要在 proxy 或 prior；
- 若 `E2b` 明显改善 `Tail-RBID`，再进入 `E2c` 的 tail-aware weighting 更有依据。
