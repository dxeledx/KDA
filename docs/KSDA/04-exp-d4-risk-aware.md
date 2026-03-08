# Exp-D.4: KCAR驱动的风险感知对齐

**前置条件**: Exp-D.3成功 (性能反馈有效)
**目标**: 整合KCAR，实现完整的风险感知动态对齐
**优先级**: 🔥 最高 (完整KSDA框架)
**预计时间**: 1-2周

---

## 1. 实验目的

整合前面所有工作，回答：

> 用KCAR控制对齐强度，结合Koopman空间对齐和性能反馈，能否实现最优的动态适应？

**核心假设**:
- KCAR量化了"源域动力学能否解释当前目标域演化"
- 这是比几何距离更直接的对齐风险信号
- 在Koopman空间，KCAR和对齐天然统一

**这是KSDA的完整版本**

---

## 2. 方法设计

### 2.1 完整Pipeline

```
初始化:
  K_s = fit_global_koopman(source_data)
  K_t = K_s.copy()
  A_0 = learn_alignment_matrix(source_data)

在线循环 (t = 1, 2, ..., T):
  Step 1: Koopman Embedding
    z_t = tangent_project(C_t, M_s)
    ψ_t = lift_quadratic(z_t)

  Step 2: Compute KCAR (关键创新)
    # 用最近窗口计算KCAR
    window = history[-32:]
    e_src = compute_residuals(window, K_s)
    e_tgt = compute_residuals(window, K_t)
    ρ_t = (e_src - e_tgt) / (e_src + e_tgt + ε)

  Step 3: Risk-Aware Alignment
    # 根据KCAR动态调整对齐强度
    w_t = sigmoid(a - b * ρ_t)  # ρ高→w低，ρ低→w高

    # 在Koopman空间插值
    ψ'_t = (1 - w_t) * ψ_t + w_t * (A_t @ ψ_t)

  Step 4: Prediction
    y_pred_t = LDA(ψ'_t)
    p_t = softmax(LDA.decision_function(ψ'_t))

  Step 5: Performance Feedback
    acc_recent = compute_recent_accuracy()
    K_t = update_koopman_with_performance(K_t, ψ_t, acc_recent)

  Step 6: Update Alignment
    A_{t+1} = update_alignment(A_t, ψ_t, feedback_t)
```

### 2.2 KCAR驱动的门控函数

**方案A: 简单线性门控** (推荐首选)

```python
w_t = sigmoid(a - b * ρ_t + c)

# 其中:
# a: 基础对齐倾向
# b: KCAR敏感度 (b > 0)
# c: 偏置

# 性质:
# ρ_t > 0 (高风险) → w_t 低 → 弱对齐
# ρ_t < 0 (低风险) → w_t 高 → 强对齐
```

**方案B: 分段线性门控** (更灵活)

```python
if ρ_t > τ_high:
    w_t = w_min  # 高风险，最小对齐
elif ρ_t < τ_low:
    w_t = w_max  # 低风险，最大对齐
else:
    w_t = linear_interpolate(ρ_t, τ_low, τ_high, w_max, w_min)
```

**方案C: 结合几何特征** (如果A/B不够)

```python
# KCAR + 几何特征的混合门控
c_t = [ρ_t, d_tgt, sigma]  # KCAR作为主信号
w_t = sigmoid(weights @ c_t + bias)
```

### 2.3 关键设计决策

**决策1: KCAR窗口大小**
- 当前: 32 trials (与stage3保持一致)
- 可以根据实验调整

**决策2: 门控函数形式**
- 第一版: 简单线性门控 (方案A)
- 如果不够再尝试方案B/C

**决策3: 是否归一化KCAR**
- KCAR已经归一化到[-1, 1]
- 不需要额外归一化

**决策4: 更新顺序**
- K_t → KCAR → w_t → 对齐 → 预测 → 反馈 → 更新A_t
- 这个顺序最合理

---

## 3. 对照实验设计

### 3.1 Baseline

**Baseline-1: KSDA-perf** (主要对照)
- 有性能反馈，但无KCAR
- 来自Exp-D.3

**Baseline-2: 窗口级KCAR策略**
- 协方差空间 + 窗口级KCAR
- 来自stage3

**Baseline-3: RA + CSP + LDA**
- 固定对齐
- 性能: 43.4%

### 3.2 KSDA-full变体

**KSDA-full-A**: 简单线性KCAR门控
**KSDA-full-B**: 分段线性KCAR门控
**KSDA-full-C**: KCAR + 几何混合门控

**消融变体**:
- **KSDA-full-no-kcar**: 无KCAR，用几何特征
- **KSDA-full-no-perf**: 有KCAR，无性能反馈
- **KSDA-full-no-online**: 有KCAR，但A_t和K_t固定

---

## 4. 评估指标

### 4.1 性能指标

**主指标**:
- LOSO balanced accuracy
- 必须 > KSDA-perf
- 目标: > RA baseline (43.4%)

**次指标**:
- 逐被试准确率
- 最差被试改善幅度
- Session后期性能

### 4.2 KCAR有效性

**KCAR与性能的关系**:
- ρ_t与当前trial准确率的相关性
- ρ_t与未来窗口性能的预测能力

**KCAR与对齐强度的关系**:
- ρ_t与w_t的相关性
- 验证"ρ高→w低"的逻辑

### 4.3 完整性指标

**与stage3的对比**:
- KSDA-full vs 窗口级KCAR策略
- 证明Koopman空间对齐的优势

**计算效率**:
- 在线推理时间
- 必须 < RA baseline

---

## 5. 理论分析

### 5.1 为什么KCAR在Koopman空间更有效？

**统一性**:
- KCAR本身就在Koopman空间计算
- 对齐也在Koopman空间
- 风险评估和对齐在同一表示空间

**一致性**:
- KCAR量化动力学一致性
- 对齐保持动力学结构
- 两者目标一致

### 5.2 完整KSDA的理论保证

**Regret Bound**:
- 对齐矩阵A_t: O(√T) (from Exp-D.2)
- Koopman算子K_t: O(√T) (from Exp-D.3)
- 门控权重w_t: O(√T) (在线凸优化)

**收敛性**:
- 在适当条件下，(A_t, K_t, w_t) → (A*, K*, w*)

---

## 6. 实验协议

### 6.1 数据集
- BNCI2014001 (9个被试)
- LOSO交叉验证

### 6.2 超参数

**KCAR门控**:
- a (基础对齐倾向): 1.0
- b (KCAR敏感度): 2.0
- c (偏置): 0.0
- KCAR窗口: 32 trials

**其他参数**:
- 继承Exp-D.2和D.3的最优参数

### 6.3 评估协议

**在线评估**:
- 记录ρ_t, w_t, 准确率
- 分析三者关系

**离线评估**:
- 最终LOSO准确率
- 与所有baseline对比

---

## 7. 实现计划

### 7.1 代码结构

```
src/alignment/ksda_full.py
  - KSDAFull类
  - 整合所有组件
  - compute_kcar_gate()

experiments/ksda_exp_d4.py
  - 主实验脚本
  - 完整KSDA pipeline

src/evaluation/ksda_full_analysis.py
  - KCAR-性能关系分析
  - 完整性验证
```

### 7.2 实现步骤

**Week 1**:
1. 整合KCAR计算 (2天)
2. 实现KCAR门控 (2天)
3. 整合所有组件 (1天)

**Week 2**:
1. 运行LOSO实验 (2天)
2. 完整分析和可视化 (3天)

---

## 8. 输出文件

### 8.1 结果目录
```
results/ksda/exp_d4/<run_name>/
  - loso_results.csv
  - comparison_all_methods.csv
  - summary.json
  - kcar_analysis/
      - kcar_vs_performance.pdf
      - kcar_vs_weight.pdf
      - risk_aware_alignment.pdf
  - details/
      - subject_Axx.npz (完整历史)
```

### 8.2 必须包含的字段

**comparison_all_methods.csv**:
- 包含所有方法的对比:
  - RA baseline
  - KSDA-static (D.1)
  - KSDA-online (D.2)
  - KSDA-perf (D.3)
  - KSDA-full (D.4)
  - 窗口级KCAR策略 (stage3)

---

## 9. 验收标准

### 9.1 最小成功标准 (必须满足)

1. **性能提升**:
   - KSDA-full > KSDA-perf + 1%
   - KSDA-full ≥ RA baseline (43.4%)

2. **KCAR有效性**:
   - ρ_t与w_t负相关 (验证逻辑正确)
   - ρ_t能预测性能变化

3. **优于stage3**:
   - KSDA-full > 窗口级KCAR策略

### 9.2 理想成功标准 (期望达到)

1. **显著超越baseline**:
   - KSDA-full > RA + 3%
   - 在7/9个被试上优于RA

2. **最差被试改善**:
   - Subject 2性能 > 30% (当前19.8%)

3. **计算效率**:
   - 在线推理时间 < RA
   - 快5-10倍

### 9.3 失败判定

1. **性能无提升**:
   - KSDA-full ≈ KSDA-perf
   - KCAR未带来额外价值

2. **性能下降**:
   - KSDA-full < KSDA-perf
   - KCAR引入负面影响

3. **不如stage3**:
   - KSDA-full < 窗口级KCAR策略
   - Koopman空间对齐无优势

---

## 10. 失败应对方案

### 如果KCAR无效

**可能原因**:
1. KCAR窗口太小或太大
2. 门控函数形式不合适
3. KCAR与几何特征冲突

**修复尝试**:
- 调整KCAR窗口大小
- 尝试分段线性门控
- 尝试KCAR + 几何混合

### 如果不如stage3

**可能原因**:
1. Koopman空间对齐本身不如协方差空间
2. 在线更新引入不稳定
3. 复杂度过高

**应对**:
- 回到Exp-D.1分析失败原因
- 考虑简化版本
- 或接受stage3的窗口级策略

---

## 11. 成功后的下一步

### 如果达到最小成功标准

**立即行动**:
1. 写Exp-D.4 results memo
2. 完整消融研究
3. 准备论文Method section

### 如果达到理想成功标准

**加速推进**:
1. 立即开始多数据集验证 (Exp-D.5)
2. 撰写论文初稿
3. 准备投稿

---

## 12. 论文定位

### 如果KSDA-full成功

**Title**:
```
Koopman-Space Dynamic Alignment for Cross-Subject Motor Imagery BCI
```

**核心贡献**:
1. 首次在Koopman空间做跨被试对齐
2. 性能驱动的Koopman算子在线修正
3. KCAR驱动的风险感知动态适应
4. 统一的理论框架和实证验证

**目标会议**:
- NeurIPS 2026 (理论创新)
- ICML 2027 (Koopman理论)

### 如果部分成功

**调整策略**:
- 强调Koopman空间对齐的效率优势
- 诚实报告KCAR的局限性
- 目标AAAI 2027或JNE

---

## 13. 关键问题清单

在开始实现前，需要明确:

- [ ] KCAR门控函数形式 (推荐简单线性)
- [ ] KCAR窗口大小 (32 trials)
- [ ] 门控参数 (a, b, c)
- [ ] 是否需要归一化
- [ ] 更新顺序
- [ ] 如何处理边界情况 (ρ_t极端值)

---

## 14. 与前面实验的关系

**递进关系**:
- D.1: 证明Koopman空间对齐可行
- D.2: 证明在线更新A_t有效
- D.3: 证明性能反馈修正K_t有效
- D.4: 整合KCAR，完整KSDA框架

**如果D.4失败**:
- D.1-D.3的工作仍然有价值
- 可以发表部分结果
- 或回到stage3的窗口级策略

---

**状态**: 📝 规划完成，等待Exp-D.3成功
**前置条件**: Exp-D.3 ✅
**预计开始**: Exp-D.3完成后
**预计完成**: Exp-D.3完成后1-2周
