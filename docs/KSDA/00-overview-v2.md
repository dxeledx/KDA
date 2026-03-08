# KSDA实验路线图 v2.0 (基于Exp-B.1洞察的重大调整)

**更新日期**: 2026-03-08
**更新原因**: Exp-B.1证明问题不在尺度,在信号本身
**核心调整**: 优先验证KCAR作为更好的信号,而非继续优化几何特征

> **执行状态更新**  
> 本文档的 `D.1 / D.1+` 现已冻结为旧 Phase 1 参考。  
> 当前真正执行的新主线已经切换为：`D0 → D1 → D2 → D3 → D4`。  
> `D1-R` 已完成，但它现在只保留为校准参考，正式结果见：`09-exp-d1r-results-memo.md`。  
> v3 对应执行文档见：`10-exp-d0-dyn-feasibility.md`、`11-exp-d1-dyn-local-experts.md`、`12-exp-d2-linear-performance-proxy.md`、`13-exp-d3-operator-correction.md`、`14-exp-d4-online-dyn-alignment.md`。

---

## 🔴 关键洞察 (来自Exp-B.1)

### Exp-B.1告诉我们什么

**不应该说**: "Exp-B失败只是因为量纲不匹配"

**应该说**: "量纲不匹配是问题之一,但归一化后gate虽然更有结构,却没有变得更正确"

### 证据

| 指标 | Exp-B | Exp-B.1 | 变化 |
|------|-------|---------|------|
| w_std | 0.0057 | 0.0881 | ✅ +15倍 (结构增强) |
| accuracy | 41.36% | 40.74% | ❌ -0.62% (性能下降) |
| corr(w, d_src) | 0.48 | 0.48 | → 仍然d_src主导 |

**结论**:
- ✅ 归一化修复了gate塌缩
- ❌ 但gate有结构 ≠ gate做正确决策
- 🔴 **问题不在尺度,在信号本身**

### 问题的层次

```
Level 1: 量纲问题 ✅ 已解决 (归一化)
  ↓
Level 2: 信号问题 ❌ 未解决 (几何特征不是因果预测器)
  ↓
Level 3: 性能提升 ❌ 未达到
```

**当前卡在Level 2**: gate会动了,但动得不对

---

## 为什么KSDA更有希望了

### 几何特征的根本局限

**几何特征 (d_src, d_tgt, sigma)**:
- 只告诉你"当前trial在哪里"
- 不告诉你"对齐后会怎样"
- 归一化改变了尺度,但没改变信号的本质

**学到的模式可能是错的**:
- 当前: d_src高 → w_t高 (gate学到的)
- 可能应该: d_src高 → w_t低 (离源域远,对齐风险高)
- 归一化让这个错误模式更明显了

### KCAR的优势

**KCAR是直接的风险信号**:
```
几何特征: "当前trial在哪里" (间接,需要学习映射)
KCAR: "源域动力学能否解释当前演化" (直接,无需学习)
```

**在Koopman空间,KCAR和对齐天然统一**:
- KCAR本身就在Koopman空间计算
- 对齐也在Koopman空间
- 不需要学习"几何→风险"的间接映射

---

## 🔥 调整后的实验路线

### 原计划 (已废弃)

```
D.1 (静态对齐) → D.2 (在线更新A_t) → D.3 (性能反馈K_t) → D.4 (KCAR)
```

**问题**:
- D.2/D.3仍然在优化几何特征的使用
- Exp-B.1已经证明这个方向有根本局限

### 新计划 (v2.0)

```
D.1 (静态对齐) → D.1+ (KCAR门控,A/K固定) → [决策点] → D.2/D.3 (如果需要)
```

**理由**:
- 优先验证KCAR这个"更好的信号"
- 避免在几何特征上继续打转
- 如果KCAR有效,再考虑在线更新

---

## Phase 1: Exp-D.1 (静态Koopman对齐) - 不变

**目标**: 验证Koopman空间对齐的基础可行性

**方法**:
```python
ψ_t = lift(tangent_project(C_t))
ψ'_t = A @ ψ_t  # 固定对齐矩阵
y_pred = LDA(ψ'_t)
```

**验收标准**:
- ✅ 性能 ≥ RA baseline (43.4%)
- ✅ 计算时间 < RA

**时间**: 1-2周

---

## Phase 2: Exp-D.1+ (KCAR门控,A/K固定) - 🆕 新增

**目标**: 验证KCAR作为信号的有效性,避免重蹈Exp-B.1覆辙

**方法**:
```python
# 计算KCAR
ρ_t = compute_kcar(K_s, K_t, window=32)

# KCAR驱动的门控 (简单版)
w_t = sigmoid(a - b * ρ_t + c)  # b > 0

# 在Koopman空间插值
ψ'_t = (1 - w_t) * ψ_t + w_t * (A @ ψ_t)

# 分类
y_pred = LDA(ψ'_t)
```

**关键**: A、K_s、K_t 都是固定的,只测试 KCAR 作为信号

### 对比实验

**Baseline-1**: KSDA-static (from D.1)
- 固定A, w=1.0
- 性能: 预期≈43.4%

**Baseline-2**: KSDA-geometric-gate (Exp-B.1在Koopman空间)
- 固定A, 用归一化后的几何特征预测w_t
- 这是对Exp-B.1的直接对比

**KSDA-kcar-gate** (新方案):
- 固定A, 用KCAR预测w_t
- 直接对比KCAR vs 几何特征

### 验收标准

**最小成功**:
- KSDA-kcar-gate > KSDA-geometric-gate + 1%
- 证明KCAR是更好的信号

**理想成功**:
- KSDA-kcar-gate > KSDA-static (RA baseline)
- 证明KCAR驱动的动态对齐有效

**失败判定**:
- KSDA-kcar-gate ≈ KSDA-geometric-gate
- 说明KCAR在trial-level也无效

### 时间

**1-2周** (在D.1完成后)

---

## 决策点: D.1+的结果决定后续路线

### 情况A: KCAR明显优于几何特征 ✅

**证据**:
- KSDA-kcar-gate > KSDA-geometric-gate + 2%
- KCAR与性能有明显相关性

**行动**:
→ 继续Phase 3: Exp-D.2 (在线更新A_t)
→ 继续Phase 4: Exp-D.3 (性能反馈K_t)
→ 完整KSDA框架

**预期**:
- 完整KSDA成功
- 冲击NeurIPS 2026

---

### 情况B: KCAR略优于几何特征 🟡

**证据**:
- KSDA-kcar-gate > KSDA-geometric-gate + 0.5-1%
- 但仍未超越RA baseline

**行动**:
→ 不继续D.2/D.3 (在线更新可能无法弥补差距)
→ 转向**窗口级KCAR策略**

**窗口级策略**:
```python
# 不是trial-level连续控制
# 而是窗口级离散决策

if ρ_window > threshold_high:
    action = "use w=0.5"  # 高风险,部分对齐
elif ρ_window < threshold_low:
    action = "use w=1.0"  # 低风险,完全对齐
else:
    action = "use w=0.7"  # 中等风险
```

**预期**:
- 窗口级策略成功
- 投AAAI 2027

---

### 情况C: KCAR不优于几何特征 ❌

**证据**:
- KSDA-kcar-gate ≈ KSDA-geometric-gate
- KCAR与性能无明显相关

**行动**:
→ 停止trial-level动态控制这条线
→ 回到stage3的窗口级KCAR诊断
→ 或考虑改变对齐方法本身

**深入分析**:
1. 为什么KCAR也无效?
   - 是Koopman算子估计不准?
   - 还是trial-level控制本身就很难?

2. 窗口级KCAR是否仍有效?
   - 回到stage3的结果 (AUROC=0.69)
   - 深化窗口级分析

**预期**:
- 论文定位调整为"KCAR作为诊断工具"
- 投AAAI 2027或JNE

---

## Phase 3-4: 在线更新 (条件执行)

**仅在情况A时执行**

### Phase 3: Exp-D.2 (在线更新A_t)

**前置条件**: D.1+ 成功 (KCAR明显优于几何特征)

**方法**:
```python
# 在线更新对齐矩阵
A_{t+1} = A_t - η * grad(A_t)

# 仍用KCAR门控
w_t = sigmoid(a - b * ρ_t + c)
ψ'_t = (1 - w_t) * ψ_t + w_t * (A_t @ ψ_t)
```

**验收标准**:
- KSDA-online > KSDA-kcar-gate + 1%

**时间**: 2-3周

---

### Phase 4: Exp-D.3 (性能反馈K_t)

**前置条件**: D.2 成功

**方法**:
```python
# 根据性能加权更新K_t
K_t = update_with_performance_feedback(K_t, acc_recent)

# 用更新后的K_t计算KCAR
ρ_t = compute_kcar(K_s, K_t, window)
```

**验收标准**:
- KSDA-perf > KSDA-online + 1%

**时间**: 2-3周

---

## 更新后的时间线

### 乐观情况 (情况A)

```
Week 1-2:  Exp-D.1 ✅
Week 3-4:  Exp-D.1+ ✅ (KCAR明显优于几何特征)
Week 5-7:  Exp-D.2 ✅
Week 8-10: Exp-D.3 ✅
Week 11+:  多数据集验证 + 论文撰写
```

**目标**: NeurIPS 2026 (5月截止)

---

### 中等情况 (情况B)

```
Week 1-2:  Exp-D.1 ✅
Week 3-4:  Exp-D.1+ 🟡 (KCAR略优)
Week 5-6:  窗口级KCAR策略 ✅
Week 7-8:  多数据集验证
Week 9+:   论文撰写
```

**目标**: AAAI 2027 (8月截止)

---

### 保守情况 (情况C)

```
Week 1-2:  Exp-D.1 ✅
Week 3-4:  Exp-D.1+ ❌ (KCAR无效)
Week 5-6:  深化stage3窗口级分析
Week 7-8:  失败原因分析
Week 9+:   论文撰写 (负结果+诊断工具)
```

**目标**: AAAI 2027或JNE

---

## 更新后的决策树

```
Exp-D.1 (静态Koopman对齐)
├─ 成功 → Exp-D.1+ (KCAR门控,A/K固定)
│  ├─ 情况A: KCAR明显优于几何特征
│  │  └─ D.2 (在线更新A_t) → D.3 (性能反馈K_t)
│  │     └─ 完整KSDA成功 → 冲击NeurIPS 2026
│  │
│  ├─ 情况B: KCAR略优于几何特征
│  │  └─ 窗口级KCAR策略
│  │     └─ 投AAAI 2027
│  │
│  └─ 情况C: KCAR不优于几何特征
│     └─ 回到stage3深化分析
│        └─ 投AAAI 2027或JNE (诊断工具)
│
└─ 失败 → 回到stage3
   └─ 深化窗口级KCAR分析
```

---

## 风险分析更新

### 风险1: Koopman空间对齐失败 (不变)

**概率**: 30%
**应对**: 见原RISKS-AND-SOLUTIONS.md

---

### 风险2: KCAR在trial-level也无效 (🆕 新增)

**概率**: 30-40%

**具体表现**:
- KSDA-kcar-gate ≈ KSDA-geometric-gate
- ρ_t与性能无明显相关
- w_t的变化与性能改善无关

**根本原因**:

**原因1: Trial-level噪声太大**
- 单个trial的KCAR估计不稳定
- 需要更大的窗口

**原因2: Koopman算子估计不准**
- K_s和K_t都是噪声估计
- 它们的残差差异也是噪声

**原因3: 连续控制本身就很难**
- 最优w_t可能是离散的
- 连续预测引入不必要的复杂性

**应对方案**:

**方案A: 增大KCAR窗口**
```python
# 当前: window=32
# 尝试: window=64, 128
```

**方案B: 改为离散决策**
```python
# 不预测连续w_t
# 改为离散动作
if ρ_t > 0.3:
    w_t = 0.5
elif ρ_t < -0.3:
    w_t = 1.0
else:
    w_t = 0.7
```

**方案C: 转向窗口级策略**
```python
# 不是trial-level
# 而是每N个trial决策一次
ρ_window = compute_kcar(window=32)
action = decide_action(ρ_window)  # 离散动作
```

**方案D: 止损退出**
- 承认trial-level动态控制很难
- 回到stage3的窗口级KCAR诊断
- 论文定位为"KCAR作为诊断工具"

---

### 风险3: 时间不够 (更新)

**概率**: 20% (降低了)

**原因**: 新路线更短
- 如果D.1+失败,可以快速转向窗口级
- 不需要完成D.2/D.3

---

## 论文叙事更新

### 如果情况A (完整KSDA成功)

**Title**: Koopman-Space Dynamic Alignment for Cross-Subject Motor Imagery BCI

**核心贡献**:
1. 识别了几何特征的根本局限 (Exp-B.1的价值)
2. 提出KCAR作为直接的风险信号
3. 在Koopman空间实现统一的动态对齐框架
4. 理论保证 + 实证验证

**Exp-B.1的角色**:
- 重要的负结果
- 证明了问题不在尺度,在信号
- 为KCAR的必要性提供了证据

**目标**: NeurIPS 2026

---

### 如果情况B (窗口级成功)

**Title**: Risk-Aware Cross-Subject Alignment via Koopman Operator Analysis

**核心贡献**:
1. KCAR作为有效的风险诊断器
2. 窗口级策略的合理性
3. Koopman空间对齐的效率优势

**Exp-B.1的角色**:
- 证明了trial-level连续控制的挑战
- 为窗口级策略提供了理由

**目标**: AAAI 2027

---

### 如果情况C (KCAR也失败)

**Title**: Understanding the Challenges of Dynamic Alignment in Cross-Subject BCI

**核心贡献**:
1. 系统性分析了动态对齐的三个层次问题
   - 量纲问题 (已解决)
   - 信号问题 (部分解决)
   - 方法问题 (仍存在)
2. KCAR作为诊断工具的价值
3. 负结果的学术价值

**Exp-B.1的角色**:
- 关键的诊断实验
- 帮助理解问题的本质

**目标**: AAAI 2027或JNE

---

## 立即行动 (本周)

### 1. 开始实现Exp-D.1

**优先级**: 🔥 最高

**任务**:
- 实现Koopman embedding
- 实现对齐矩阵学习
- 运行LOSO实验

**时间**: 5-7天

---

### 2. 同时准备D.1+的代码

**优先级**: 🔥 高

**任务**:
- 复用stage3的KCAR计算代码
- 实现简单的KCAR门控
- 准备对比实验框架

**时间**: 2-3天 (与D.1并行)

---

### 3. 准备Plan B (窗口级策略)

**优先级**: 🟡 中

**任务**:
- 设计窗口级KCAR策略的详细方案
- 准备实验代码框架

**时间**: 1-2天

---

## 成功标准 (更新)

### 最小成功 (可发表)

- Exp-D.1成功 (Koopman空间对齐 ≥ RA)
- Exp-D.1+证明KCAR优于几何特征
- 窗口级策略有效
- **论文**: AAAI 2027

---

### 中等成功 (好论文)

- Exp-D.1成功
- Exp-D.1+成功 (KCAR明显优于几何特征)
- Exp-D.2成功 (在线更新有效)
- **论文**: AAAI 2027或JNE

---

### 完全成功 (顶会)

- Exp-D.1-D.3全部成功
- 完整KSDA框架
- 性能超越RA baseline 3%+
- **论文**: NeurIPS 2026或ICML 2027

---

## 文档更新记录

- `00-overview.md`: 已更新 (本文档)
- `01-exp-d1-static-alignment.md`: 不变
- `02-exp-d2-online-update.md`: 标记为"条件执行"
- `03-exp-d3-performance-feedback.md`: 标记为"条件执行"
- `04-exp-d4-risk-aware.md`: 拆分为D.1+ (简化版)
- `ROADMAP.md`: 需要更新
- `RISKS-AND-SOLUTIONS.md`: 需要更新 (新增风险2)

---

**当前状态**: 📌 v3 主线已落地，下一步先做 `D0`
**关键变化**: 不再先问“静态器够不够强”，而是先问“数据里是否存在值得动态化的局部决策结构”
**下一步**: 运行 `D0`，只有它证明动态需求存在，才继续 `D1`
**对应文档**: `09-exp-d1r-results-memo.md`、`10-exp-d0-dyn-feasibility.md`、`11-exp-d1-dyn-local-experts.md`、`12-exp-d2-linear-performance-proxy.md`、`13-exp-d3-operator-correction.md`、`14-exp-d4-online-dyn-alignment.md`
