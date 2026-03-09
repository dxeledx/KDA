# Notes: KSDA 当前研究进展综合

## 已确认的主线
- 当前不可偏离的目标不是继续堆 controller，而是：
  1. 定义并验证脑电表征—行为不一致性；
  2. 在 Koopman 特征空间做对齐；
  3. 再讨论用线性性能指标修正 Koopman 算子。
- 这条主线已经在 `docs/KSDA/00-overview-v2.md`、`docs/KSDA/22-primary-goal-mismatch-first.md`、`docs/KSDA/KSDA.md` 中反复固定。

## 已有关键证据
- Stage 2 已经说明：连续动态 gating + 行为反馈不是当前瓶颈解法，论文叙事转向 `RA-first / safe adaptation`。
- `D1-T rank=48` 成功把动作空间拉开，说明表征瓶颈真实存在，动作集本身已可用。
- `D1.5` 失败说明当前瓶颈不是动作，而是 teacher / supervisory signal 的构造。
- `RBID` 已能支撑“问题存在且现有方法仍存在该问题”。
- `K-RBID` 目前只有中等强度的局部诊断价值，暂不适合直接驱动 online update。

## 当前最自然的 paper-facing 主实验
- 先做一个保守的 `RBID-aware Koopman aligner`：
  - 静态或窗口级；
  - 目标是降低 `RBID / Tail-RBID` 并提高 `LOSO accuracy`；
  - 不把 `teacher signal`、`trial-level selector`、`online controller` 绑进主实验。

## 当前最重要的实验设计原则
- 每一轮只改一个主因素。
- 所有方法统一报告：`accuracy / kappa / RBID / Tail-RBID`。
- 如果新方法只能改善几何相似性，不能改善行为排序或 accuracy，则不能算主线进展。
- 若 `K-RBID` 仍然不稳定，则保留为诊断工具，不强行方法化。

## E0 刷新结果（2026-03-09）
- 运行目录：`results/e0/2026-03-09-e0-r1`
- 主表已经固定在：`results/e0/2026-03-09-e0-r1/summary/e0_main_table.csv`
- 当前主表中：
  - `RA` 仍是最高 LOSO accuracy：`0.4375`
  - `EA` 与 `RA` 拿到最低 `RBID`：`0.2817`
  - `Koopman-noalign-r48`：`accuracy=0.3943`, `RBID=0.3413`
  - `Static Koopman aligner-r48`：`accuracy=0.3835`, `RBID=0.3135`
- 这说明：
  - 当前 paper-facing Koopman baseline 仍明显落后于 `EA/RA`
  - 但 `Static Koopman aligner-r48` 的 mismatch 已低于 `Koopman-noalign-r48`
  - 因此 `E1` 最自然的 control 是 `Static Koopman aligner-r48`
- 旧 baseline 缓存与 `E0` 刷新结果存在明显漂移；后续统一以 `E0` 归档结果为准。

## E1 结果（2026-03-09）
- 运行目录：`results/e1/2026-03-09-e1-r1`
- 方法：`Conservative Koopman aligner-r48`
- 相对 `Static Koopman aligner-r48`：
  - `accuracy_mean`: `0.4248 vs 0.3835`，提升 `+0.0413`
  - `rbid`: `0.3056 vs 0.3135`，下降 `-0.0079`
  - `tail_rbid`: `0.5503 vs 0.6357`，下降 `-0.0854`
- 这意味着：
  - 光靠“保守残差对齐 + source 判别保持 + 动力学一致性”，不加 `RBID surrogate`，已经能让当前 Koopman 线明显更稳
  - 当前主 idea 的顺序站住了：先做保守对齐，再把 mismatch-aware supervision 加进去
- 因此 `E2` 的唯一主改动应当是：**在 E1 相同参数化上加入 `RBID surrogate`**，其余保持不变。

## E2 结果（2026-03-09）
- 运行目录：`results/e2/2026-03-09-e2-r1`
- 方法：`RBID-aware Conservative Koopman aligner-r48`
- 设计：
  - 行为先验：`E0/RA pairwise transfer accuracy`
  - 相似度代理：`aligned source-block mean` vs `aligned target-train mean` cosine
  - 唯一新项：`L_rank`
- 相对 `E1`：
  - `accuracy_mean`: `0.4232 vs 0.4248`，下降 `-0.0015`
  - `rbid`: `0.3095 vs 0.3056`，变差 `+0.0040`
  - `tail_rbid`: `0.6234 vs 0.5503`，明显变差 `+0.0731`
- 这说明：
  - 当前 `RBID surrogate` 第一版**没有站住**
  - 问题不在“加 ranking supervision 这件事本身一定错”，而更可能在：
    - 行为先验选取不合适；
    - 相似度代理太弱；
    - surrogate 把优化推向了对 RBID 指标并不真实有利的方向
- 因此下一步不应直接进入 `E3 / E4`，而应先在 `E2` 内部做 surrogate 诊断与重设计。

## E2-Diag 结果（2026-03-09）
- 运行目录：`results/e2_diag/2026-03-09-e2diag-r1`
- 目的：
  - 排除一个关键混杂：`E1` 与 `E2` 的 pairwise mismatch 评估协议并不一致。
  - 用与 `E2` 完全相同的 `target-global pooled-source per target` 协议，重算一次 `lambda_rank=0` 的 conservative baseline。
- refreshed conservative baseline（同 E2 协议）：
  - `pairwise_accuracy_mean = 0.3208`
  - `RBID = 0.2937`
  - `Tail-RBID = 0.6190`
  - `Pearson-r = 0.3579`
- 相对 `E2`：
  - `pairwise_accuracy_mean`: `0.3208 vs 0.3149`
  - `RBID`: `0.2937 vs 0.3095`
  - `Tail-RBID`: `0.6190 vs 0.6234`
- 这说明：
  - 之前的 `E1/E2` 比较确实有 protocol mismatch，后续不能再直接引用原始 pairwise 对比。
  - 但即便去掉这个混杂，当前 `RBID surrogate` 版本仍然没有优于 refreshed conservative baseline。
  - 因此主问题不只在评估口径，更在当前 surrogate / similarity proxy 的设计。
- 当前更可信的下一步是：
  - 先固定 pairwise 协议；
  - 保持 `RA prior` 不动；
  - 优先替换当前 `aligned mean cosine` similarity proxy。

## E2-ProxyDiag 结果（2026-03-09）
- 运行目录：`results/e2_proxy_diag/2026-03-09-e2proxy-r1`
- 目的：
  - 在固定 `RA prior` 与 `target-global pooled-source per target` 协议下，只诊断 proxy 本身。
- 当前训练 proxy：`proxy_train_mean_cosine`
  - `RBID_vs_behavior = 0.4286`
  - `mean_target_corr_behavior_spearman = -0.2436`
  - 对真实 behavior 呈明显负相关，说明它本身就在把排序往错误方向推。
- 最优 behavior-facing proxy：`proxy_test_mean_neg_l2`
  - `RBID_vs_behavior = 0.2738`
  - `mean_target_corr_behavior_spearman = 0.3974`
- 最优 RA-facing proxy：`proxy_test_cka`
  - `RBID_vs_ra = 0.2738`
  - `mean_target_corr_ra_spearman = 0.4020`
- 这说明：
  - 当前主瓶颈已经基本锁定在 **similarity proxy**，而不是 `RA prior`。
  - train-mean cosine 这条线应当停止继续投入。
  - 下一步最合理的是做一个 **proxy replacement only** 版本，先保持 `RA prior` 不变。

## E2a 结果（2026-03-09）
- 运行目录：`results/e2a/2026-03-09-e2a-r1`
- 改动：
  - 保持 `RA prior`、target-global pairwise protocol、当前 pairwise logistic ranking loss 不变；
  - 只把 training proxy 从 `aligned mean cosine` 换成 `mean+dyn` 结构化 score。
- 结果：
  - `LOSO accuracy = 0.4228`，低于 `E1` 的 `0.4248`
  - `pairwise RBID = 0.3095`，高于 refreshed baseline 的 `0.2937`
  - `pairwise Tail-RBID = 0.6327`，高于 refreshed baseline 的 `0.6190`
  - `pairwise accuracy mean = 0.3146`，低于 refreshed baseline 的 `0.3208`
- 这说明：
  - `aligned mean cosine` 确实不是好 proxy，但当前 `mean+dyn` 替换版也没有把 E2 救回来。
  - 因此下一步不应继续做“只换 proxy”的路线，而应进入 `E2b`：优先修改 surrogate 形式本身。

## E2b 结果（2026-03-09）
- 运行目录：`results/e2b/2026-03-09-e2b-r1`
- 改动：
  - 保持 `RA prior`、`mean+dyn` proxy、target-global pairwise protocol 不变；
  - 只把 mismatch surrogate 从 `pairwise logistic` 换成 `Soft-RBID + Huber`。
- 结果：
  - `LOSO accuracy = 0.4213`，低于 `E1` 的 `0.4248`
  - `pairwise RBID = 0.3095`，高于 refreshed baseline 的 `0.2937`
  - `pairwise Tail-RBID = 0.6234`，高于 refreshed baseline 的 `0.6190`
  - `pairwise accuracy mean = 0.3145`，低于 refreshed baseline 的 `0.3208`
- 相对 `E2a`：
  - `RBID` 没有改善
  - `Tail-RBID` 略有回落（`0.6327 -> 0.6234`）
  - `pairwise accuracy mean` 仍略降
- 这说明：
  - surrogate 形式本身确实影响了一部分 tail 行为，但目前仍不足以把方法拉回到 refreshed baseline 之上。
  - 当前更像是：`proxy / prior / surrogate` 三者耦合问题，而不是某一个单点替换就能解决。

## E2c 结果（2026-03-09）
- 运行目录：`results/e2c/2026-03-09-e2c-r1`
- 改动：
  - 保持 `RA prior`、`mean+dyn` proxy、`Soft-RBID + Huber` 不变；
  - 只加 `tail-aware weighting`，默认 `rank_tail_weight = 2.0`、`rank_tail_quantile = 0.25`。
- 结果：
  - `LOSO accuracy = 0.4228`，仍低于 `E1` 的 `0.4248`
  - `pairwise RBID = 0.3095`，仍高于 refreshed baseline 的 `0.2937`
  - `pairwise Tail-RBID = 0.6234`，与 `E2b` 持平
  - `pairwise accuracy mean = 0.3145`，仍低于 refreshed baseline 的 `0.3208`
- 相对 `E2b`：
  - `RBID` 不变
  - `Tail-RBID` 不变
  - `pairwise accuracy mean` 只有极小变化
- 这说明：
  - 当前默认 tail weighting 没有提供额外收益；
  - 到 `E2c` 为止，单独改 proxy、单独改 surrogate、单独加 tail weighting 都不够。
  - 研究上更合理的下一步应回到 `behavior prior + representation proxy` 组合本身，而不是继续堆 extension 层实验。
