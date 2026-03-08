# Exp-B 结果备忘录：Trial-Level 动态 Gate 几何版

**日期**: 2026-03-07  
**实验目录**: `results/stage3/trial_dynamic_gate_exp_b/2026-03-07-trial-dynamic-exp-b`  
**实验定位**: 在 `Exp-A` 的基础上，将 gate 输入从 `d_tgt` 扩展为 `d_src + d_tgt + sigma_recent`，验证 richer geometry 是否能让 `w_t` 不再塌成近常数，并推动性能更接近强默认策略 `fixed w=1.0 / RA`。

---

## 1. 实验目的

Exp-B 要回答的问题只有一个：

> richer geometry 是否真的能把 trial-level dynamic gate 从“方向正确但近常数”的状态，推进到“结构更丰富且性能更强”的状态？

因此本轮重点不在于最终平均性能本身，而在于三件事：

1. `w_t` 是否比 `Exp-A` 更有结构；
2. `w_t` 与几何信号之间是否形成更合理的关系；
3. `dynamic_exp_b` 是否开始逼近甚至局部超过 `fixed w=1.0`。

---

## 2. 主要结果

### 2.1 平均性能

| 方法 | mean accuracy | std |
|------|---------------|-----|
| fixed `w=1.0` | **0.4340** | 0.1406 |
| dynamic_exp_b | **0.4136** | 0.1389 |
| fixed `w=0.5` | 0.4101 | 0.1352 |
| fixed `w=0.0` | 0.3804 | 0.1179 |

### 2.2 与 Exp-A 的直接比较

Exp-B 相对 Exp-A：

- `accuracy_mean_delta = -0.00116`
- `wins_vs_exp_a = 3`
- `losses_vs_exp_a = 4`
- `w_std_mean_delta = +0.00011`

结论很直接：

> Exp-B 没有形成对 Exp-A 的稳定改进。

它既没有把平均性能明显推高，也没有把 `w_t` 的结构显著拉开。

---

## 3. 被试级现象

### 3.1 dynamic_exp_b vs fixed `w=1.0`

dynamic_exp_b 相对 `fixed w=1.0`：

- 更好：`A04`
- 更差：`A01`, `A02`, `A03`, `A05`, `A06`, `A07`, `A08`, `A09`

这说明 Exp-B 依然**不具备稳定挑战强默认策略 RA 的能力**。

### 3.2 dynamic_exp_b vs fixed `w=0.5`

dynamic_exp_b 相对 `fixed w=0.5`：

- 更好：`A01`, `A03`, `A04`, `A06`, `A07`, `A08`
- 更差：`A02`, `A05`, `A09`

这说明 dynamic gate 仍然不是完全没用；它依然比常数 `w=0.5` 更聪明一些。  
但这个优势依旧不够强，无法转化成对 `fixed w=1.0` 的系统性追近。

---

## 4. `w_t` 结构分析

### 4.1 有信息的一面

从细节文件看：

- `corr(w, d_src) > 0`：`9/9` 个被试
- `corr(w, d_tgt) > 0`：`7/9`
- `corr(w, sigma_recent) > 0`：`6/9`

这说明 gate 并没有学坏，它在利用几何信息调节 `w_t`。

### 4.2 关键问题：`w_t` 仍然近常数

Exp-B 的核心失败点不在“方向错误”，而在“幅度不够”：

- `w_mean = 0.5916`
- `w_std_mean = 0.00570`

与 `Exp-A` 对比：

- `Exp-A w_std_mean ≈ 0.00559`
- `Exp-B w_std_mean ≈ 0.00570`

只增加了约 `0.00011`。

这意味着：

> Exp-B 不是把 gate 变得更有结构，而是把 gate 的整体均值往更高的对齐强度抬了一截。

换句话说，Exp-B 更像：

> “一个围绕 `0.59` 波动的近常数 gate”

而不是：

> “一个能根据 trial 结构做明显调节的动态 gate”

---

## 5. 当前最合理的解释

我认为 Exp-B 的主要问题不是“假设错了”，而是**输入尺度与主导关系有问题**。

### 5.1 `d_src` 很可能主导了 gate

从被试级均值看：

- `d_src_mean ≈ 0.251`
- `d_tgt_mean ≈ 0.113`
- `sigma_mean ≈ 0.0097`

三者量级差异明显。

而 gate 当前是简单线性层：

`w_t = sigmoid(a·d_src + b·d_tgt + c·sigma_recent + bias)`

在没有特征归一化的情况下，最自然的结果就是：

- 数值更大的 `d_src` 抢走主导权
- `sigma_recent` 因量级太小几乎没有话语权
- `d_tgt` 的信息被部分淹没

这和观察结果吻合：

- `corr(w, d_src)` 最稳定
- `w_mean` 被整体推高
- 但 `w_std` 没有明显增加

### 5.2 richer geometry 没有转化成 richer structure

这轮最重要的负结论是：

> “多加了几何特征” ≠ “trial-level gate 自动变得更细粒度”。

当前它只带来了：

- 更高的平均 `w`
- 但没有带来更大的 trial-level 分辨率

---

## 6. 这轮可以下什么结论

### 可以下的结论

1. **trial-level 动态 gate 仍然值得继续。**
   - 它没有崩
   - 它仍然优于 `fixed w=0.0`
   - 多数情况下也优于 `fixed w=0.5`

2. **Exp-B 说明单纯堆几何特征还不够。**
   - richer geometry 自身没有自动把 gate 拉出近常数状态

3. **当前最值得优先修的是特征尺度，而不是继续加更复杂控制器。**

### 不能下的结论

1. 不能说 Exp-B 已经优于 Exp-A
2. 不能说几何增强已经证明有效
3. 不能说现在应该直接进入 `Exp-C` 加 `rho_window`

---

## 7. 是否继续到 Exp-C？

### 我的判断：**现在不建议直接进入 Exp-C**

原因很明确：

- `w_t` 仍然几乎是常数
- Exp-B 没有稳定优于 Exp-A
- 如果现在直接加 `rho_window`，很容易把问题混在一起：
  - 到底是 `rho_window` 真有用
  - 还是几何输入本身就没校准好

更合理的顺序应该是：

## **Exp-B.1：归一化修正版**

先只修一件事：

- 对 `d_src / d_tgt / sigma_recent` 做 feature-wise 归一化或尺度校准

再看：

1. `w_std_mean` 能否明显高于 Exp-A / Exp-B；
2. `corr(w, d_tgt)` 能否重新变成主导关系；
3. `dynamic` 平均性能能否至少不低于 Exp-A。

如果 Exp-B.1 依然不行，再谈 Exp-C 才更有意义。

---

## 8. 当前最合理的下一步

当前阶段三最合理的推进顺序已经变得更清楚：

1. **Exp-A**：证明最小 trial-level gate 方向正确  
2. **Exp-B**：发现 richer geometry 直接叠加无效，且存在尺度主导问题  
3. **Exp-B.1**：先修输入尺度 / 归一化  
4. **Exp-C**：只有在 trial-level gate 本身站稳后，再把 `rho_window` 当作慢变量注入  
5. **Exp-D**：最后才接回最简单 feedback

---

## 9. 正式判断

Exp-B 的价值不在于“拿到了更强结果”，而在于：

> 它帮助我们确认了当前 bottleneck 并不在“还缺一个更复杂信号”，而更可能在“gate 输入的尺度与主导关系没有处理好”。  

因此，Exp-B 是一次**有价值的负结果**：

- 它不支持直接进 Exp-C
- 但它非常明确地指向了一个更小、更干净、更该先做的下一步：**Exp-B.1（归一化修正版）**
