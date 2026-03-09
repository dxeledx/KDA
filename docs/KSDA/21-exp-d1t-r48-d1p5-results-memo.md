# D1-T (rank=48) + D1.5 正式结果备忘录

**日期**: 2026-03-09  
**对应运行**:
- `results/ksda/exp_d1t/2026-03-08-ksda-d1t-r48-r1`
- `results/ksda/exp_d1p5/2026-03-09-ksda-d1p5-r48-r3`

---

## 一句话结论

这轮结果给出了一个比之前更清晰的分层结论：

> **提高 Koopman 表征维度到 `pca_rank=48` 后，trial-safe 二维动作集终于被“拉开”了；但把 window oracle 直接因果化成 trial-level teacher signal 之后，这个 teacher 仍然明显不如最佳单动作。**

换句话说：
- **动作空间问题**：已经明显改善
- **window → trial 因果蒸馏问题**：仍然不成立

因此当前阶段最合理的决策不是立刻进入 `D2 / D3`，而是先重新设计 `D1.5` 的老师信号。

---

## 1. D1-T（rank=48）回答了什么

### 核心结果

`D1-T rank=48` 的主结果是：

- `best_single_action = P2_diag_scaling_a100`
- `best_single_accuracy_mean = 0.4614`
- `high_overlap_actions = []`
- `trial_safe_action_space_valid = true`

这相对之前的 `rank=16` 是一个实质性变化：

| 版本 | best single acc | high-overlap actions | gate |
|------|------------------|----------------------|------|
| `D1-T rank=16` | `0.4572` | `5` | `fail` |
| `D1-T rank=48` | `0.4614` | `0` | `pass` |

### 这说明什么

1. **表征瓶颈确实存在**
   - 之前动作集“过挤”并不只是动作原语设计问题
   - `pca_rank=48` 让动作在决策层明显解挤

2. **trial-safe 二维动作集方向是成立的**
   - 最优单动作已稳定落在 `P2_diag_scaling_a100`
   - 动作空间终于不再被 `identity` 主导

3. **现在可以进入 `D1.5`**
   - 因为当前动作空间已经满足“至少值得做 causal trialization”的前提

所以对 `D1-T` 这一步，当前最准确的结论是：

> **动作空间本身已经可用；当前新的主要不确定性不在动作集，而在老师信号的构造方式。**

---

## 2. D1.5 回答了什么

### 核心结果

`D1.5` 的 summary 是：

- `teacher_window_size = 16`
- `teacher_accuracy_mean = 0.4255`
- `best_single_action = P2_diag_scaling_a100`
- `best_single_accuracy_mean = 0.4614`
- `window_oracle_accuracy_mean = 0.5166`
- `mean_delta_vs_best_single = -0.0359`
- `wins_vs_best_single = 0`
- `teacher_vs_window_oracle_agreement = 0.1728`
- `causal_trialization_valid = false`

### 这说明什么

1. **teacher signal 远弱于 window oracle**
   - `window oracle = 0.5166`
   - `teacher = 0.4255`
   - 这说明“上一完整窗口最优动作”在当前窗口并不能稳定延续

2. **teacher 甚至不如最佳单动作**
   - `teacher - best_single = -0.0359`
   - `wins = 0/9`
   - 所以当前 teacher signal 不能被当作合理的 trial-level 上游标签

3. **teacher 与原 window oracle 的一致率极低**
   - `0.1728`
   - 这不是“小幅退化”，而是结构性错位

因此 `D1.5` 的结论很明确：

> **“上一完整窗口的 oracle action” 不是一个可用的 causal trial-level teacher。**

---

## 3. 什么是“老师信号”，它从哪里来

你刚才问得非常关键。

### 什么是老师信号

在当前这条线里，**老师信号（teacher signal）** 指的是：

> 一个我们暂时不打算直接学习出来、而是先人为构造的“上游动作标签”，用来告诉后续模型：在当前时刻，什么动作更像是应该选的。

它的角色类似于：
- 监督学习里的 pseudo-label
- 行为克隆里的 teacher policy
- 蒸馏里的 teacher output

也就是说，`D2` 并不是直接从最终 accuracy 反推动作，而是先试图去拟合一个“老师告诉你的动作标签”。

### 它从哪里来

在这次 `D1.5` 里，老师信号来自：

1. 先在 `window=16` 上算 **non-causal window oracle action**
2. 得到每个窗口的最优动作
3. 再把 **上一完整窗口** 的最优动作，赋给当前窗口中的每个 trial
4. 第一个窗口统一用 `A0_identity`

所以这次的老师信号，本质上是：

> **由 window oracle 派生出来的、因果右移一窗后的 pseudo-action labels**

### 为什么这次老师信号失败了

因为它默认了一个很强的假设：

> “上一个窗口最优的动作，很可能也是当前窗口每个 trial 的合理动作。”

而这轮结果说明，这个假设在当前数据上并不成立。

因此，当前失败的不是：
- 动作空间
- trial-safe 专家定义

而是：
- **从 window oracle 到 trial-level teacher 的桥接方式**

---

## 4. 这轮结果合在一起说明了什么

把 `D1-T rank=48` 和 `D1.5` 合在一起看，当前最重要的诊断是：

> **我们已经把动作空间问题基本拆开了，现在卡住的是“teacher signal 设计”，不是“trial-safe 动作是否存在”。**

更具体地说：

1. `D1-T` 证明：
   - 只要表征足够，trial-safe 动作集是可以被拉开的

2. `D1.5` 证明：
   - 不能把 window-level oracle 直接因果右移，当成 trial-level teacher

所以当前最合理的研究口径是：

> **当前问题不再是“动作空间不够好”，而是“我们还没有找到合适的 trial-level teacher / supervisory signal”。**

---

## 5. 这轮可以支持什么，不能支持什么

### 可以支持的结论

1. `pca_rank=48` 明显缓解了 Koopman 表征导致的动作挤压问题
2. trial-safe 二维动作集已经可以作为后续研究的基础
3. 直接使用“上一完整窗口 oracle action”作为 trial-level teacher 不成立

### 不能支持的结论

1. 不能说当前 `D1.5` 已经成功桥接了 window → trial
2. 不能说现在就应该直接进入 `D2 / D3`
3. 不能说当前线性 proxy 的失败风险来自动作空间本身

---

## 6. 正式决策

### 本轮决策

**不进入 `D2 / D3`。**

### 原因

不是因为动作空间还不够好，而是因为：

> **当前 teacher signal 设计不成立。**

如果现在直接进 `D2`，那你学到的只会是一个去拟合错误老师标签的 selector。

---

## 7. 对下一步的建议

这轮最该做的不是重新动动作集，而是重做 `D1.5`。

优先方向建议：

### 方向 A：重做 teacher signal

下一版应优先尝试：
- 不再用“上一完整窗口最优动作”
- 改成更细粒度、但仍严格因果的 teacher 构造方式

例如：
- trailing teacher
- change-point aware teacher
- confidence-filtered teacher
- 只在动作切换点附近给 teacher，而不是整窗常数标签

### 方向 B：动作空间先保持不动

因为现在 `D1-T rank=48` 已经说明：
- 动作集终于被拉开了
- 不宜立刻回去再动 `P1/P2/P3/P4`

当前最小改动、最高信息量的下一步，就是：

> **固定 `D1-T rank=48` 动作集，只重做 `D1.5` 的老师信号。**

---

## 最终判断

如果只保留一句最重要的话：

> **`D1-T rank=48` 成功说明动作空间问题已明显缓解；`D1.5` 失败说明真正的下一步瓶颈不在动作，而在 teacher signal 的构造。**
