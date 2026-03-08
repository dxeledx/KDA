# 阶段三窗口级 KCAR / RA-first policy 结果备忘录

**日期**: 2026-03-07  
**对应实验**:
- `results/stage3/kcar_diagnostic/2026-03-07-kcar-diagnostic-v2`
- `results/stage3/kcar_policy/2026-03-07-kcar-policy-v1`

---

## 1. 当前已经得到的可靠结论

### 1.1 KCAR 作为窗口级风险信号是成立的

从 `kcar_diagnostic-v2` 看：

- `mean_auroc = 0.6877`
- `mean_auprc = 0.7201`
- `mean_spearman = 0.2727`

这说明 `KCAR` 不是噪声信号，而是能够在窗口级区分：

- 哪些窗口更适合保持 `RA`
- 哪些窗口偏离 `RA` 更可能有利

### 1.2 RA-first 二动作策略是对的

将动作空间限制为：

- `stay_with_ra (w=1.0)`
- `use_partial_alignment (w=0.5)`

是合理的。  
结果显示：

- 在低偏离预算下，允许少量窗口偏离 `RA` 的策略有机会超过纯 `RA`
- 偏离预算一旦变大，`RA` 又会重新变成更稳的默认策略

这与“RA 是强默认策略，偏离应当稀疏且保守”的判断一致。

### 1.3 KCAR policy 已经开始显示出新增价值

在 `near-causal`、低 coverage（尤其 `0.1~0.2`）下：

- `KCAR policy > d_tgt policy > RA`

这说明：

- `KCAR` 不只是“有信息”
- 它已经开始转化成比最强简单几何启发式更好的窗口选择

---

## 2. 当前还不能下的结论

### 2.1 不能说 KCAR 已经是稳定主控制器

原因：

- `KCAR` 的优势目前只在部分 coverage 区间成立
- `worst-subject delta` 仍然为负
- 一旦 coverage 变大，`RA` 经常重新成为最优

因此现阶段更准确的表述应是：

> `KCAR` 是一个有效的窗口级风险信号，且在低偏离预算下开始体现新增决策价值，但还不足以被宣称为稳定主控制器。

### 2.2 不能把窗口级结果直接等同于最终 online 结论

当前所有策略实验仍然是：

- 窗口级
- budget-matched
- surrogate action selection

它们还没有真正回答：

> trial-level 在线动态 gate 是否能稳定地学出有意义的 `w_t`，并在因果条件下优于固定基线。

### 2.3 不能说这组结果已经足以支撑顶会/顶刊主结论

原因包括：

1. 仍是单数据集
2. 仍是窗口级 surrogate，而非原始的 trial-level online 主设定
3. `KCAR` 对 `d_tgt` 的优势还不够全面、最差被试仍未压稳
4. 还没有形成“从风险诊断 → trial-level 动态 gate”的完整因果证据链

---

## 3. 为什么“窗口级结果 ≠ 最终 trial-level 在线结论”

你的最初设想是：

> 以 trial 为单位，在线决定当前需要多强对齐。

而当前窗口级实验做的是：

> 用窗口级 risk score 去判断某些窗口是否值得偏离 `RA`。

这两者的关系是：

- **窗口级结果**：回答某个 risk signal 值不值得进入控制链路
- **trial-level 结果**：回答这个 risk signal 进入控制链路后，是否能真正形成有效在线控制

所以窗口级结果只是：

> **桥接证据（bridging evidence）**

而不是最终证据。

它能支持下一步做 trial-level online gate，但不能替代它。

---

## 4. 为什么这些结果仍然值得保留

尽管不能直接写成最终论文主结果，这批结果仍然非常重要，因为它们已经帮助我们完成了两件事：

### 4.1 排除了错误主线

我们已经不需要再把 `KCAR` 当作一个新的窗口级主控制器继续扩写。  
更合理的位置是：

- 先把它降为“桥接证据”
- 再把它作为下一轮 trial-level gate 的慢变量候选

### 4.2 缩小了下一轮实验搜索空间

当前最值得继续的方向非常清楚：

1. 回到最小原型，直接做 **trial-level 动态 gate**
2. 第一轮只用最简单几何信号（如 `d_tgt`）
3. 如果它本身站得住，再把 `rho_window` 作为慢变量注入
4. 最后才把最简单 feedback 接回来

因此这批结果的正确用途不是“写进最终论文主表”，而是：

> 作为阶段三的正式迭代记录和下一轮实验的证据起点。

---

## 5. 对当前阶段的正式判断

### 可以说

- `KCAR` 作为窗口级风险信号成立
- `RA-first` 二动作策略成立
- 在低预算、near-causal 条件下，`KCAR policy` 已经开始优于最强简单几何对手 `d_tgt`

### 不可以说

- `KCAR` 已经是成熟主控制器
- trial-level online 动态对齐已经被证明成立
- 当前结果已足以支撑顶会/顶刊的最终主结论

### 最合理的下一步

> 回到最小 trial-level 原型，验证动态 `w_t` 本身是否站得住；若成立，再把窗口级 `rho_window` 当作慢变量注入 gate。
