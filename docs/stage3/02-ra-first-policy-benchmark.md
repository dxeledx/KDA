# RA-first 二动作策略 Benchmark 设计

## 1. 实验目标

这一步不是证明 `KCAR` 已经可以当主控制器，而是验证：

> 在同样“允许偏离 RA”的预算下，`KCAR policy` 是否比最强简单对照更会挑窗口。

因此主比较不是：

- `KCAR vs RA`

而是：

- `KCAR policy vs d_tgt policy vs sigma_recent policy vs RA`

---

## 2. 动作空间

主策略固定为二动作：

- `stay_with_ra` → `w = 1.0`
- `use_partial_alignment` → `w = 0.5`

`w = 0.0` 仅用于第一阶段风险诊断定义 oracle 标签，不进入 policy 主结果。

---

## 3. 双 setting

### 3.1 retrospective

当前窗口 `j` 的 score 可以使用该窗口的完整信息。

这是上界，用来回答：

> 如果 risk signal 信息最完整，它是否比简单 heuristic 更好？

### 3.2 near-causal

当前窗口 `j` 的动作只能使用历史窗口信息。

第一版实现采用：

- 历史窗口得分可见
- 当前窗口只使用上一时刻可用 score 的近因果近似

这个版本用来回答：

> 从分析 setting 走向可部署 setting 时，收益会损失多少？

---

## 4. 预算匹配

对每个 policy 和 setting，统一跑：

`coverage ∈ {0.1, 0.2, ..., 0.9}`

解释：

- `coverage = 0.3` 表示只允许 `30%` 的窗口偏离 `RA`
- 每个被试内部按 score 排序，取 top-k 窗口执行 `w=0.5`
- 其余窗口固定使用 `RA`

为什么必须预算匹配：

- 否则某个 policy 可能只是“偏离得更多”
- 我们真正要比较的是：**同样的偏离预算下，谁更会挑窗口**

---

## 5. 正文主结果

主表至少报告：

1. `mean accuracy`
2. `mean balanced accuracy`
3. `mean macro-F1`
4. `late-session accuracy`
5. `worst-subject delta vs RA`
6. `wins / losses vs d_tgt policy`

主图至少报告：

1. `coverage_performance_curve`
2. `coverage_worst_subject_curve`
3. `retrospective_vs_causal_gap`

---

## 6. 三种结果的解释模板

### 情况 A：`KCAR policy > d_tgt policy > RA`

这是最强结果。

可解释为：

- `KCAR` 不仅能做风险诊断
- 它还能比最强简单几何启发式更好地挑窗口
- `RA-first safe adaptation` 主线成立

### 情况 B：`KCAR policy ≈ d_tgt policy > RA`

这是可接受结果。

可解释为：

- 窗口级风险选择是成立的
- 但 `KCAR` 的新增价值还不够大
- 下一步应考虑细化 source Koopman，而不是推翻整条主线

### 情况 C：`d_tgt policy > KCAR policy > RA`

这也不是坏结果。

更合理的结论是：

- `KCAR` 仍然有信息
- 但还没有稳定超过最强简单启发式
- 说明当前最优下一步是细化 `source Koopman`，而不是继续加复杂 feedback 或更复杂 control

---

## 7. 当前阶段的底线

这一阶段的最低目标不是“证明 KCAR 已经足够强”，而是：

1. `KCAR policy` 至少不差于 `RA`
2. `KCAR policy` 要与 `d_tgt policy` 做公平强对照
3. near-causal setting 下不能彻底塌掉

只要这三点成立，阶段三主线就可以继续推进。
