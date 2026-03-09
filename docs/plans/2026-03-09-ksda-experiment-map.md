# KSDA Experiment Map

**Date**: 2026-03-09  
**Scope**: `BNCI2014001` cross-subject MI, current KSDA mainline  
**Role**: 作为后续实验的总控 map；凡是不清楚是否该做的实验，都先回看这份文档。

---

## 1. 主问题

当前主问题固定为：

> **先定义并验证脑电表征—行为不一致性，再设计一个保守的 Koopman 空间对齐器，让表征排序更接近行为排序，并带来更高的 cross-subject transfer performance。**

换成实验语言，就是三层目标：

1. **现象层**：问题是否存在，而且现有方法是否仍然存在这个问题？
2. **方法层**：是否能构造一个对齐器，在 Koopman 空间里同时改善行为相关性与分类表现？
3. **扩展层**：只有在前两层成立后，才讨论局部诊断驱动或算子修正。

---

## 2. 当前已经锁定的结论

### 2.1 已完成并可信的结论

- Stage 2 已明确否定“连续动态 gating + 行为反馈”作为当前主解法。
- `D1-T rank=48` 说明动作空间与表征容量问题已经被部分拆开：
  - `best_single_action = P2_diag_scaling_a100`
  - `trial_safe_action_space_valid = true`
- `D1.5` 明确失败：
  - 当前 teacher signal 不能作为合理的上游监督
  - 因此当前**不进入** `D2 / D3`
- `RBID` 已可作为主 mismatch 指标。
- `K-RBID` 目前只够做局部诊断量，不够直接驱动在线更新。

### 2.2 这些结论对后续实验的约束

- **不要**回到“先做 controller 再解释”的路线。
- **不要**把 `teacher signal / trial selector / online controller` 当作近期主实验。
- **优先**设计一个保守的、paper-facing 的 Koopman 对齐算法。

---

## 3. 论文主线版本（当前推荐）

### 3.1 主故事

1. 跨被试 MI 中存在稳定的 **representation-behavior mismatch**。
2. 现有 static baselines 与当前 Koopman 线都还没有消除该 mismatch。
3. 因此应设计一个 **mismatch-aware Koopman aligner**，而不是继续依赖不稳定的在线动作选择。
4. 一个好的对齐器应同时满足：
   - 保持源域判别性；
   - 保持或约束动力学一致性；
   - 让对齐后的表征排序更接近行为排序；
   - 在 LOSO 上改善 accuracy，并降低 `RBID / Tail-RBID`。

### 3.2 当前 paper-facing 方法原型

先做：

> **Conservative RBID-aware Koopman Aligner**

而不是：

- dynamic action selector
- trial-level online controller
- teacher-distilled causal policy

---

## 4. 方法设计边界

### 4.1 第一阶段允许做的事情

- 静态对齐器
- 窗口级对齐器
- 保守的低秩修正
- `RBID surrogate` 排序约束
- 动力学一致性约束
- 与 `RA / EA / static Koopman aligner` 的直接对比

### 4.2 第一阶段暂时不做的事情

- trial-level online gating
- teacher signal 蒸馏
- 因果 action policy 学习
- 复杂 meta-learning / RL 更新器
- 把 `K-RBID` 直接接成 online controller 的驱动量

---

## 5. 指标面板

### 5.1 主指标

- `LOSO accuracy`
- `Cohen's kappa`
- `RBID`
- `Tail-RBID`

### 5.2 次指标

- `RBID+ / RBID-`
- `Pearson(s_rep, s_beh)` 仅作辅助，不作核心结论
- 计算开销 / 参数量 / 训练稳定性

### 5.3 诊断指标

- `K-RBID`
- operator residual statistics
- 失败 pair / 失败 subject 分析

---

## 6. 实验梯度图（按顺序推进）

### E0 — 指标锁定与基线刷新

**问题**  
统一报告体系下，当前 baselines 的 mismatch 水平到底是多少？

**做什么**
- 统一刷新以下方法的结果与汇总：
  - `NoAlign`
  - `EA`
  - `RA`
  - `Static Koopman aligner`
  - `Koopman-noalign`

**输出**
- 一张统一主表：`accuracy / kappa / RBID / Tail-RBID`
- 一个 pair-level 错位分析

**通过条件**
- 指标与目前文档结论一致
- 结果可作为后续所有新方法的比较底座

---

### E1 — Conservative Koopman Aligner（不含 mismatch surrogate）

**问题**  
单靠“保守对齐 + 动力学一致性 + 判别保持”能否比当前 static Koopman aligner 更稳？

**做什么**
- 学一个保守对齐矩阵 `A = I + Δ`
- 先不加 `RBID surrogate`
- 只用：
  - 分类保持项
  - 动力学一致性项
  - 保守正则项

**目的**
- 分离“保守对齐器”本身的收益
- 避免一开始把所有 loss 混在一起

**输出**
- 与 `Static Koopman aligner` 的一对一对比

**通过条件**
- accuracy 不低于当前 static Koopman aligner
- `RBID` 至少有下降趋势

**失败解释**
- 若 accuracy 与 RBID 都无改善，则问题可能不在对齐器参数化，而在 mismatch 监督本身

---

### E2 — RBID-aware Aligner（加入 surrogate）

**问题**  
把“表征排序逼近行为排序”的约束写进训练目标后，是否能真正降低 mismatch 并提升 transfer？

**做什么**
- 在 `E1` 基础上加入 `RBID surrogate`
- surrogate 形式采用 pairwise ranking / listwise ranking 均可，但必须满足：
  - 对固定 target，比对 source-source 的可迁移性顺序
  - 行为上更可迁移的 source，在 aligned representation 上也应更接近

**输出**
- 与 `E1` 对比
- 与 `RA / EA / Static Koopman aligner` 对比

**主验证**
- `RBID ↓`
- `Tail-RBID ↓`
- `accuracy ↑`

**通过条件**
- 相比 `E1` 与现有 static baselines，至少在 `RBID / Tail-RBID` 上显著改善
- 若 accuracy 同时改善或至少不退化，可视为论文主线成立

---

### E3 — 窗口级 Conservative Variant（可选）

**问题**  
如果静态 aligner 已经有效，轻量的窗口级更新是否还能进一步改善而不破坏主线？

**做什么**
- 保持同一个 paper-facing aligner
- 只允许窗口级、保守的更新
- 禁止引入 teacher-distillation 与 trial-level selector

**定位**
- 这是增强实验，不是论文主结果前提

**进入条件**
- 只有当 `E2` 已经站住时才做

---

### E4 — Operator Correction（第二阶段）

**问题**  
当 mismatch-aware alignment 已成立后，用线性性能指标修正 Koopman 算子是否还能继续提升？

**做什么**
- 只在 `E2` 成功后进入
- 研究性能加权的算子估计或判别式 Koopman 修正

**注意**
- 这不是近期主任务
- 如果 `E2` 还没站住，禁止提前进入

---

## 7. 每轮实验必须回答的四个问题

1. 这轮改动服务于哪一层目标：现象、方法、还是扩展？
2. 这轮只改变了哪一个主因素？
3. 这轮是否同时报告了 `accuracy + RBID + Tail-RBID`？
4. 如果结果失败，它更支持“对齐器不对”还是“监督信号不对”？

若答不清，说明实验偏题。

---

## 8. 明确的 stop / go gate

### Go to E2
- `E1` 至少比当前 static Koopman aligner 更稳
- 或者虽 accuracy 未提升，但 `RBID / Tail-RBID` 已明显下降

### Go to E3
- `E2` 已能形成清楚主结果：
  - mismatch 改善
  - accuracy 不退化，最好提升

### Go to E4
- `E2` 已成为可写论文的主算法
- 需要进一步强化“性能修正算子”的第二贡献

### Stop and rethink
- 若 `E2` 连续 2 轮都只能改善几何相似性，不能改善行为排序或 accuracy
- 若 `RBID surrogate` 降了训练 loss，但 `RBID` 指标本身不降

---

## 9. 记录规则

后续每轮实验都必须在台账里记录：

- 实验 ID
- 核心假设
- 相比上一轮唯一改变了什么
- 运行目录
- 主指标结果
- 结论：继续 / 回退 / 分叉

严禁只记“感觉这轮更好”。

---

## 10. 当前最近的 3 个动作

1. 先补一版统一的 baseline 主表，作为 `E0` 固定底座。
2. 设计并写下 `E1` 的方法草图：变量、损失、训练协议、对照实验。
3. 在台账里为 `E1` 开第一条记录，后续所有 run 按台账更新。

---

## 11. 一句话提醒

> **近期主任务不是做更聪明的 controller，而是做一个更可信、更保守、能降低 mismatch 的 Koopman 对齐器。**
