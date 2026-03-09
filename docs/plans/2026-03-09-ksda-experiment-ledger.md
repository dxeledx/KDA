# KSDA Experiment Ledger

**Purpose**: 逐轮记录实验假设、改动、运行目录、结果与决策。  
**Rule**: 每做一轮实验，先登记计划，再补结果；不允许只保留运行目录而没有解释。

---

## 1. 实验注册表

| Exp ID | Status | Layer | Core Question | Single Change | Control | Main Metrics | Run Dir | Decision |
|---|---|---|---|---|---|---|---|---|
| E0 | done | phenomenon | 统一报告下 baselines 的 mismatch 到底多大？ | 全量刷新 baseline + phenomenon + 双 rank Koopman 汇总 | 现有文档结论 | acc / kappa / RBID / Tail-RBID | `results/e0/2026-03-09-e0-r1` | `E1` control 固定为 `Static Koopman aligner-r48` |
| E1 | done | method | 保守 Koopman aligner 本身是否更稳？ | 低秩残差保守对齐 + `L_cls + L_dyn + L_reg` | `Static Koopman aligner-r48` | acc / kappa / RBID / Tail-RBID | `results/e1/2026-03-09-e1-r1` | Gate 通过，可进入 `E2` |
| E2 | planned | method | 加入 RBID surrogate 后是否进一步降低 mismatch？ | 在 E1 上加 ranking surrogate | E1 | acc / kappa / RBID / Tail-RBID | TBD | TBD |
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

## 4. 使用规范

- 新实验开始前，先在“实验注册表”里更新 `Status` 和 `Single Change`。
- 跑完后必须补“主结果”和“结论”。
- 若实验失败，也必须写清楚失败更像是：
  - 对齐器参数化问题；
  - mismatch surrogate 问题；
  - 数据/协议噪声问题。
- 若一个实验不能清楚归到 `phenomenon / method / extension` 三层之一，则先不要跑。
