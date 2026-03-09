# KSDA Experiment Ledger

**Purpose**: 逐轮记录实验假设、改动、运行目录、结果与决策。  
**Rule**: 每做一轮实验，先登记计划，再补结果；不允许只保留运行目录而没有解释。

---

## 1. 实验注册表

| Exp ID | Status | Layer | Core Question | Single Change | Control | Main Metrics | Run Dir | Decision |
|---|---|---|---|---|---|---|---|---|
| E0 | done | phenomenon | 统一报告下 baselines 的 mismatch 到底多大？ | 全量刷新 baseline + phenomenon + 双 rank Koopman 汇总 | 现有文档结论 | acc / kappa / RBID / Tail-RBID | `results/e0/2026-03-09-e0-r1` | `E1` control 固定为 `Static Koopman aligner-r48` |
| E1 | done | method | 保守 Koopman aligner 本身是否更稳？ | 低秩残差保守对齐 + `L_cls + L_dyn + L_reg` | `Static Koopman aligner-r48` | acc / kappa / RBID / Tail-RBID | `results/e1/2026-03-09-e1-r1` | Gate 通过，可进入 `E2` |
| E2 | done | method | 加入 RBID surrogate 后是否进一步降低 mismatch？ | 在 E1 上加 `RBID surrogate` ranking loss | E1 | acc / kappa / RBID / Tail-RBID | `results/e2/2026-03-09-e2-r1` | 相对 E1 未通过 gate，需要诊断 surrogate 设计 |
| E2-Diag | done | diagnosis | E2 的失败有多少来自 pairwise 协议混杂？ | 用 E2 的 target-global pairwise 协议重算 `lambda_rank=0` conservative baseline | 原始 E1 pairwise + E2 | pairwise acc / RBID / Tail-RBID / Pearson-r | `results/e2_diag/2026-03-09-e2diag-r1` | 协议混杂存在，但 E2 在同口径下仍未优于 refreshed baseline |
| E2-ProxyDiag | done | diagnosis | 当前 surrogate 真正坏在 prior 还是 proxy？ | 固定 `RA prior` 与 target-global 协议，只更换 similarity proxy 做离线诊断 | E2-Diag refreshed baseline | proxy-vs-behavior RBID / Spearman / Pearson | `results/e2_proxy_diag/2026-03-09-e2proxy-r1` | 当前 `aligned mean cosine` 明显失效，下一步应优先替换 proxy |
| E3 | backlog | extension | 窗口级保守更新是否还有增益？ | 静态 → 窗口级 | E2 | acc / RBID / Tail-RBID / stability | TBD | TBD |
| E4 | backlog | extension | 线性性能修正算子是否值得作为第二贡献？ | 算子修正 | E2 | acc / RBID / operator stats | TBD | TBD |

---

## 2. 单轮实验记录模板

复制下面模板，为每轮实验补一节。

### [Exp ID - Run Tag]

**Date**:  
**Owner**: Jason + Codex  
**Status**: planned / running / done / failed / archived

#### A. 这轮要回答什么问题

-

#### B. 核心假设

-

#### C. 相比上一轮唯一主改动

-

#### D. 固定不变的控制项

- Dataset:
- Protocol:
- Representation:
- Classifier:
- Baselines:

#### E. 运行配置

- Config:
- Seed(s):
- Run dir:

#### F. 主结果

| Metric | Value | vs Control |
|---|---:|---:|
| Accuracy |  |  |
| Kappa |  |  |
| RBID |  |  |
| Tail-RBID |  |  |

#### G. 诊断结果

- Subject-level wins/losses:
- Worst-subject delta:
- Failure pairs:
- K-RBID / residual diagnostics:

#### H. 结论

- 这轮支持什么：
- 这轮不支持什么：
- 是否进入下一轮：

#### I. 下一步动作

-

---

## 3. 决策日志

### 2026-03-09

- 锁定近期主线为 `Conservative RBID-aware Koopman Aligner`。
- `teacher signal / trial-level selector / online controller` 暂不进入主实验叙事。
- `K-RBID` 暂定为诊断量，不作为近期更新驱动力。
- 完成 `E0` 全量刷新：`results/e0/2026-03-09-e0-r1`。
- 主表口径固定为：`No Alignment / EA / RA / Koopman-noalign-r48 / Static Koopman aligner-r48`。
- 现阶段 `E1` 的 control method 固定为 `Static Koopman aligner-r48`，不再回退到 `r16`。
- `E0` 刷新后的 classical baseline 与旧缓存存在可见数值漂移，后续一律以 `E0` 归档结果为准，不再混用旧表。
- 完成 `E1`：`results/e1/2026-03-09-e1-r1`。
- `Conservative Koopman aligner-r48` 相比 control：
  - `accuracy_mean: 0.4248 vs 0.3835`
  - `rbid: 0.3056 vs 0.3135`
  - `tail_rbid: 0.5503 vs 0.6357`
- `E1` 两个 gate 都通过，因此下一步主线可进入 `E2`。
- 完成 `E2`：`results/e2/2026-03-09-e2-r1`。
- `RBID-aware Conservative Koopman aligner-r48` 相对 `E1`：
  - `accuracy_mean: 0.4232 vs 0.4248`
  - `rbid: 0.3095 vs 0.3056`
  - `tail_rbid: 0.6234 vs 0.5503`
- `E2` 相对 `E1` 三个 gate 都失败；当前不能进入 `E3 / E4`，需要先诊断 surrogate 与 similarity proxy。
- 完成 `E2-Diag`：`results/e2_diag/2026-03-09-e2diag-r1`。
- 统一 pairwise 协议到 `target-global pooled-source per target` 后：
  - refreshed conservative baseline: `RBID = 0.2937`, `Tail-RBID = 0.6190`, `pairwise accuracy mean = 0.3208`
  - E2: `RBID = 0.3095`, `Tail-RBID = 0.6234`, `pairwise accuracy mean = 0.3149`
- 因此协议混杂确实存在，但 E2 在同口径下仍未优于 refreshed baseline；问题不只在评估口径，还在当前 surrogate / similarity proxy。
- 完成 `E2-ProxyDiag`：`results/e2_proxy_diag/2026-03-09-e2proxy-r1`。
- 固定 `RA prior` 与 target-global 协议后，proxy-only 诊断显示：
  - 当前训练 proxy `proxy_train_mean_cosine` 对真实 behavior 是负相关：
    - `RBID_vs_behavior = 0.4286`
    - `mean_target_corr_behavior_spearman = -0.2436`
  - 最优 behavior proxy 是 `proxy_test_mean_neg_l2`：
    - `RBID_vs_behavior = 0.2738`
    - `mean_target_corr_behavior_spearman = 0.3974`
  - 最优 RA-facing proxy 是 `proxy_test_cka`：
    - `RBID_vs_ra = 0.2738`
    - `mean_target_corr_ra_spearman = 0.4020`
- 因此当前主问题优先级已明确：先换 proxy，再决定是否保留 `RA prior`。

---

## 3.1 已完成实验

### [E0 - 2026-03-09-e0-r1]

**Date**: 2026-03-09  
**Owner**: Jason + Codex  
**Status**: done

#### A. 这轮要回答什么问题

- 在统一协议下，经典 baseline 与 Koopman baseline 的 `accuracy / kappa / RBID / Tail-RBID` 到底是什么水平？
- 哪个版本应作为后续 `E1` 的控制组？

#### B. 核心假设

- 如果重新统一刷新 baselines，并把 Koopman 主版本切到 `rank=48`，就能得到后续实验可复用的固定底座。

#### C. 相比上一轮唯一主改动

- 没有引入新方法；只刷新 baseline、pairwise mismatch 指标和主表口径。

#### D. 固定不变的控制项

- Dataset: `BNCI2014001`
- Protocol: `LOSO` + pairwise transfer mismatch
- Representation: classical CSP/LDA；Koopman 使用 quadratic lifting
- Classifier: `LDA`
- Baselines: `No Alignment / EA / RA / Koopman-noalign / Static Koopman aligner`

#### E. 运行配置

- Config: 当前仓库默认 `configs/*.yaml`
- Seed(s): 单次 deterministic refresh
- Run dir: `results/e0/2026-03-09-e0-r1`

#### F. 主结果

| Metric | Value | vs Control |
|---|---:|---:|
| Best accuracy | `RA = 0.4375` | classical best |
| Lowest RBID | `EA = 0.2817` / `RA = 0.2817` | mismatch best |
| Koopman-noalign-r48 accuracy | `0.3943` | below RA |
| Static Koopman aligner-r48 accuracy | `0.3835` | below RA |
| Static Koopman aligner-r48 RBID | `0.3135` | better than Koopman-noalign-r48 |

#### G. 诊断结果

- Main table: `results/e0/2026-03-09-e0-r1/summary/e0_main_table.csv`
- Historical Koopman comparison: `results/e0/2026-03-09-e0-r1/summary/e0_historical_koopman_table.csv`
- Paper-facing memo: `docs/KSDA/25-e0-baseline-refresh-memo.md`
- `rank=48` 相对 `rank=16`：
  - `Koopman-noalign`: correlation 更高，`RBID` 略降，但 accuracy 下降
  - `Static Koopman aligner`: `RBID` 下降，但 accuracy 下降
- 旧 baseline 缓存与新刷新结果不完全一致，提示之前缓存已不应继续作为主引用。

#### H. 结论

- 这轮支持什么：
  - `E0` 底座已固定，后续所有方法统一对照同一张主表
  - `Static Koopman aligner-r48` 可以作为 `E1` 的直接 control
  - 当前 Koopman 线仍有明显 mismatch 改善空间
- 这轮不支持什么：
  - 不能说明 `rank=48` 现成就优于所有旧 Koopman 设置
  - 不能说明只提升表征 rank 就足以提升主结果
- 是否进入下一轮：
  - 是，进入 `E1`

#### I. 下一步动作

- 固定 `Static Koopman aligner-r48` 为 control，开始设计并实现保守 Koopman aligner（不含 mismatch surrogate）。

---

### [E1 - 2026-03-09-e1-r1]

**Date**: 2026-03-09  
**Owner**: Jason + Codex  
**Status**: done

#### A. 这轮要回答什么问题

- 在不引入 `RBID surrogate` 的前提下，单靠“保守残差对齐 + source 判别保持 + source 动力学约束”，能否比 `Static Koopman aligner-r48` 更稳？

#### B. 核心假设

- 当前 Koopman 线的问题不一定来自 mismatch supervision 缺失，也可能来自 static aligner 过于激进；如果改成 source-anchored 的低秩残差修正，可能同时保住 accuracy 并降低 mismatch。

#### C. 相比上一轮唯一主改动

- 从现有 static Koopman aligner，切换到 `Conservative Koopman aligner-r48`：
  - `A_res = I + B S B^T`
  - `B` 来自 source `A1/LDA-style` 判别子空间
  - 损失为 `L_cls + L_dyn + L_reg`

#### D. 固定不变的控制项

- Dataset: `BNCI2014001`
- Protocol: `LOSO` + pairwise transfer mismatch
- Representation: `pca_rank=48`, `quadratic`
- Classifier: `LDA`
- Control: `Static Koopman aligner-r48`

#### E. 运行配置

- Config: 当前仓库默认 `configs/*.yaml`
- Seed(s): 单次 deterministic run
- Run dir: `results/e1/2026-03-09-e1-r1`

#### F. 主结果

| Metric | Value | vs Control |
|---|---:|---:|
| Accuracy | `0.4248` | `+0.0413` |
| Kappa | `0.2330` | `+0.0550` |
| RBID | `0.3056` | `-0.0079` |
| Tail-RBID | `0.5503` | `-0.0854` |

#### G. 诊断结果

- `pairwise_scores.csv` 行数 = `360`，与 E0 主比较口径一致
- `rbid_method_comparison.csv` 已包含 `Conservative Koopman aligner-r48`
- Gate:
  - `accuracy_pass = true`
  - `rbid_pass = true`
  - `ready_for_e2 = true`
- 结果目录中的 `summary.json` 已显式记录 control 与 delta

#### H. 结论

- 这轮支持什么：
  - 保守参数化本身就能改善当前 Koopman 线
  - 不引入 `RBID surrogate` 的情况下，已经能同时提升 accuracy 并降低 mismatch
  - 当前主 idea 的“先对齐、再 mismatch-aware”顺序是合理的
- 这轮不支持什么：
  - 不能说明已经超过 `RA / EA`
  - 不能说明 `RBID surrogate` 已经不是必须的；它仍然是下一步增强的核心变量
- 是否进入下一轮：
  - 是，进入 `E2`

#### I. 下一步动作

- 在 `E1` 的保守残差对齐器上加入 `RBID surrogate`，并保持其它设置不变，做 `E2` 的唯一主因素改动。

---

### [E2 - 2026-03-09-e2-r1]

**Date**: 2026-03-09  
**Owner**: Jason + Codex  
**Status**: done

#### A. 这轮要回答什么问题

- 在 `E1` 的同一保守对齐器上，只加入 `RBID surrogate` 后，是否能进一步降低 mismatch 并至少不伤 accuracy？

#### B. 核心假设

- 若把“表征排序逼近行为排序”写进训练目标，`RBID / Tail-RBID` 应该比 `E1` 更低，并且 accuracy 至少不退化。

#### C. 相比上一轮唯一主改动

- 在 `E1` 的 `L_cls + L_dyn + L_reg` 基础上加入 `L_rank`
- 行为先验固定为 `E0/RA pairwise transfer accuracy`
- 相似度代理固定为 `aligned source-block mean` 与 `aligned target-train mean` 的 cosine

#### D. 固定不变的控制项

- Dataset: `BNCI2014001`
- Protocol: `LOSO` + pairwise transfer mismatch
- Representation: `pca_rank=48`, `quadratic`
- Base aligner: `Conservative Koopman aligner-r48`
- Static control: `Static Koopman aligner-r48`

#### E. 运行配置

- Config: 当前仓库默认 `configs/*.yaml`
- Seed(s): 单次 deterministic run
- Run dir: `results/e2/2026-03-09-e2-r1`

#### F. 主结果

| Metric | Value | vs E1 |
|---|---:|---:|
| Accuracy | `0.4232` | `-0.0015` |
| Kappa | `0.2310` | `-0.0021` |
| RBID | `0.3095` | `+0.0040` |
| Tail-RBID | `0.6234` | `+0.0731` |

#### G. 诊断结果

- 相对 static control：
  - accuracy 仍高 `+0.0397`
  - RBID 略低 `-0.0040`
  - Tail-RBID 略低 `-0.0123`
- 相对 E1：
  - `accuracy_pass = false`
  - `rbid_pass = false`
  - `tail_rbid_pass = false`
  - `ready_for_e3_like_extensions = false`
- `pairwise_scores.csv` 行数 = `432`
- `rbid_method_comparison.csv` 已包含 `RBID-aware Conservative Koopman aligner-r48`

#### H. 结论

- 这轮支持什么：
  - ranking supervision 没有把方法彻底带坏；相对 static control 仍维持正收益
  - 当前 surrogate 至少在实现上闭环可跑、指标可比
- 这轮不支持什么：
  - 当前 `RBID surrogate` 设计没有优于 `E1`
  - 当前 `behavior prior + similarity proxy` 组合不能作为最终 paper-facing 主结果
- 是否进入下一轮：
  - 不进入 `E3 / E4`
  - 先回到 `E2` 内部做 surrogate 诊断与重设计

#### I. 下一步动作

- 优先诊断两件事：
  1. `E0/RA` 作为行为先验是否过强或错配；
  2. `aligned mean cosine` 是否过弱，无法真正承载 RBID 排序目标。

---

### [E2-Diag - 2026-03-09-e2diag-r1]

**Date**: 2026-03-09  
**Owner**: Jason + Codex  
**Status**: done

#### A. 这轮要回答什么问题

- E2 相对 E1 的变差，有多少只是来自 pairwise 评估协议不一致？

#### B. 核心假设

- 如果把 `lambda_rank=0` 的 conservative baseline 放到与 E2 相同的 pairwise 协议下，原始 E1/E2 的 mismatch 比较会发生可见变化。

#### C. 相比上一轮唯一主改动

- 不改参数化、不改 representation、不改 classifier，只把保守 aligner 的 pairwise 评估协议改成与 E2 相同的 `target-global pooled-source per target`。

#### D. 固定不变的控制项

- Dataset: `BNCI2014001`
- Representation: `pca_rank=48`, `quadratic`
- Base aligner: `Conservative Koopman aligner-r48`
- Classifier: `LDA`
- Pairwise protocol for refreshed baseline: 与 E2 完全一致

#### E. 运行配置

- Run dir: `results/e2_diag/2026-03-09-e2diag-r1`

#### F. 主结果

| Metric | Value | vs E2 |
|---|---:|---:|
| Refreshed pairwise accuracy mean | `0.3208` | `+0.0060` |
| Refreshed RBID | `0.2937` | `-0.0159` |
| Refreshed Tail-RBID | `0.6190` | `-0.0043` |
| Refreshed Pearson-r | `0.3579` | `-0.0117` |

#### G. 诊断结果

- 原始 E1 pairwise 协议确实与 E2 不同，原比较存在混杂。
- 但在统一协议后，refreshed conservative baseline 仍优于 E2：
  - `RBID: 0.2937 < 0.3095`
  - `Tail-RBID: 0.6190 < 0.6234`
  - `pairwise accuracy mean: 0.3208 > 0.3149`
- target-wise 平均相关：
  - `corr_ra_vs_accuracy_spearman`: refreshed `0.5377` vs E2 `0.5012`
  - `corr_accuracy_vs_cka_spearman`: refreshed `0.3044` vs E2 `0.2567`

#### H. 结论

- 这轮支持什么：
  - 之后所有 surrogate 诊断都必须统一到同一 pairwise 协议
  - 当前 E2 的问题不只是评估口径，surrogate / proxy 本身也在拉偏
- 这轮不支持什么：
  - 不能把 E2 的失败完全归因于 protocol mismatch
- 是否进入下一轮：
  - 仍不进入 `E3 / E4`

#### I. 下一步动作

- 在统一 pairwise 协议下，优先替换 similarity proxy；`RA prior` 先保留不动。

---

### [E2-ProxyDiag - 2026-03-09-e2proxy-r1]

**Date**: 2026-03-09  
**Owner**: Jason + Codex  
**Status**: done

#### A. 这轮要回答什么问题

- 在固定 `RA prior` 和统一 pairwise 协议后，当前 E2 失败到底更像是 prior 问题，还是 proxy 问题？

#### B. 核心假设

- 如果只替换 similarity proxy 做离线诊断，当前训练 proxy 与真实 behavior 的排序一致性会显著差于更合理的候选 proxy。

#### C. 相比上一轮唯一主改动

- 不改 aligner、不改 prior、不改 protocol，只对多种 candidate proxy 做离线 ranking 诊断。

#### D. 固定不变的控制项

- Dataset: `BNCI2014001`
- Representation: `pca_rank=48`, `quadratic`
- Base aligner: `Conservative Koopman aligner-r48`
- Pairwise protocol: `target-global pooled-source per target`
- Behavior prior: `E0/RA pairwise transfer accuracy`

#### E. 运行配置

- Run dir: `results/e2_proxy_diag/2026-03-09-e2proxy-r1`

#### F. 主结果

| Metric | Value | vs Current Training Proxy |
|---|---:|---:|
| Current proxy (`proxy_train_mean_cosine`) RBID vs behavior | `0.4286` | baseline |
| Best behavior proxy (`proxy_test_mean_neg_l2`) RBID vs behavior | `0.2738` | `-0.1548` |
| Current proxy mean target Spearman vs behavior | `-0.2436` | baseline |
| Best behavior proxy mean target Spearman vs behavior | `0.3974` | `+0.6410` |

#### G. 诊断结果

- 当前训练 proxy `proxy_train_mean_cosine`：
  - 对 behavior 是负相关；
  - 对 RA prior 也几乎没有稳定正相关。
- `proxy_test_mean_neg_l2` 是最好的 behavior-facing proxy。
- `proxy_test_cka` 是最好的 RA-facing proxy。
- train-mean family 整体都弱，test-side distribution-sensitive proxy 明显更可信。

#### H. 结论

- 这轮支持什么：
  - 当前 `aligned mean cosine` 是第一优先级问题
  - proxy 的问题已经足够强，不需要先去动 `RA prior`
- 这轮不支持什么：
  - 不能说明 `RA prior` 已经完全没问题，只能说明它不是当前首要瓶颈
- 是否进入下一轮：
  - 继续在 `E2` 内部做 proxy 替换，不进入 `E3 / E4`

#### I. 下一步动作

- 先做一个 `proxy replacement only` 版本：保持 `RA prior` 不变，把 `aligned mean cosine` 换成更强的 proxy。

---

## 4. 使用规范

- 新实验开始前，先在“实验注册表”里更新 `Status` 和 `Single Change`。
- 跑完后必须补“主结果”和“结论”。
- 若实验失败，也必须写清楚失败更像是：
  - 对齐器参数化问题；
  - mismatch surrogate 问题；
  - 数据/协议噪声问题。
- 若一个实验不能清楚归到 `phenomenon / method / extension` 三层之一，则先不要跑。
