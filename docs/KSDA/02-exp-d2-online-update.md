# Exp-D.2: 在线更新对齐矩阵

**前置条件**: Exp-D.1成功 (KSDA-static ≥ RA baseline)
**目标**: 验证在线更新对齐矩阵能否进一步提升性能
**优先级**: 🔥 高
**预计时间**: 2-3周

---

## 1. 实验目的

在Exp-D.1的基础上，回答：

> 在线适应对齐矩阵A_t，是否能比固定A_0带来性能提升？

**核心假设**:
- 固定的A_0是在源域学习的，可能不完全适合目标域
- 在线更新A_t可以逐渐适应目标域的特性
- Koopman线性空间使得在线更新更高效、更稳定

---

## 2. 方法设计

### 2.1 Pipeline

```
输入: 协方差序列 {C_t}

初始化:
  A_0 = learn_alignment_matrix(source_data)  # 从Exp-D.1复用

在线循环 (t = 1, 2, ..., T):
  Step 1: Koopman Embedding
    z_t = tangent_project(C_t, M_s)
    ψ_t = lift_quadratic(z_t)

  Step 2: Dynamic Alignment
    ψ'_t = A_t @ ψ_t

  Step 3: Prediction
    y_pred_t = LDA(ψ'_t)
    p_t = softmax(LDA.decision_function(ψ'_t))

  Step 4: Online Update (关键创新)
    if should_update(p_t, history):
      A_{t+1} = update_alignment(A_t, ψ_t, feedback_t)
    else:
      A_{t+1} = A_t
```

### 2.2 在线更新算法

**方案A: 梯度下降 (OGD)** (推荐首选)

```python
# 伪标签反馈
y_pseudo = argmax(p_t)
confidence = max(p_t)

# 只在高置信度时更新
if confidence > τ_conf:
    # 计算梯度
    grad = ∇_A L_CE(LDA(A_t @ ψ_t), y_pseudo)

    # 梯度下降
    A_{t+1} = A_t - η_t * grad

    # 投影到约束集 (保持正交性或有界性)
    A_{t+1} = project(A_{t+1})
```

**方案B: 指数加权移动平均 (EMA)** (备选)

```python
# 用当前样本更新统计量
if confidence > τ_conf:
    # 更新类条件均值
    μ_y[y_pseudo] = (1-α) * μ_y[y_pseudo] + α * ψ_t

    # 重新计算对齐矩阵
    A_{t+1} = recompute_alignment(μ_y, Σ_y)
```

**方案C: 在线二阶方法 (如果A/B不够)**

```python
# 维护二阶信息 (类似Adam)
m_t = β1 * m_{t-1} + (1-β1) * grad
v_t = β2 * v_{t-1} + (1-β2) * grad²

A_{t+1} = A_t - η * m_t / (√v_t + ε)
```

### 2.3 关键设计决策

**决策1: 何时更新**
- 选项A: 每个trial都更新 (可能不稳定)
- 选项B: 只在高置信度时更新 (推荐)
- 选项C: 每N个trial更新一次 (batch update)
- **选择B**: 平衡适应性和稳定性

**决策2: 学习率调度**
- 选项A: 固定学习率 η
- 选项B: 递减学习率 η_t = η_0 / √t
- 选项C: 自适应学习率 (Adam-style)
- **选择B**: 理论保证更好

**决策3: 正则化**
- L2正则: ||A_t - A_0||²_F (不要偏离初始太远)
- 正交约束: A_t^T A_t = I (保持正交性)
- **选择**: L2正则 + 软正交约束

**决策4: 置信度阈值**
- τ_conf = 0.7 (初始值)
- 可以根据在线性能动态调整

---

## 3. 对照实验设计

### 3.1 Baseline

**Baseline-1: KSDA-static** (主要对照)
- 固定A_0，不更新
- 来自Exp-D.1

**Baseline-2: RA + CSP + LDA**
- 协方差空间对齐
- 性能: 43.4%

### 3.2 KSDA-online变体

**KSDA-online-A**: OGD更新，高置信度触发
**KSDA-online-B**: EMA更新，高置信度触发
**KSDA-online-C**: Batch更新，每32个trial

**消融变体**:
- **KSDA-online-no-reg**: 无正则化
- **KSDA-online-always**: 每个trial都更新
- **KSDA-online-low-conf**: 低置信度阈值(0.5)

---

## 4. 评估指标

### 4.1 性能指标

**主指标**:
- LOSO balanced accuracy
- 必须 > KSDA-static

**次指标**:
- 在线准确率曲线 (随trial变化)
- Session前期 vs 后期性能
- 最差被试改善幅度

### 4.2 适应性指标

**对齐矩阵变化**:
- ||A_t - A_0||_F 随时间变化
- A_t的谱变化 (特征值分布)

**更新频率**:
- 实际触发更新的trial比例
- 不同被试的更新频率差异

### 4.3 稳定性指标

**性能方差**:
- 在线准确率的标准差
- 是否出现性能突降

**数值稳定性**:
- A_t的条件数
- 是否出现NaN或Inf

---

## 5. 理论分析

### 5.1 Regret Bound

**在线凸优化框架**:

假设:
1. 损失函数L_t(A)对A是凸的
2. ||A||_F ≤ B (有界)
3. ||∇L_t(A)||_F ≤ G (Lipschitz)

则OGD算法的regret:
```
Regret_T = Σ L_t(A_t) - min_A Σ L_t(A)
         ≤ (B² + G²Σ η_t²) / (2η_t)
         = O(√T)  (当 η_t = 1/√t)
```

**实际意义**:
- 平均regret: O(1/√T) → 0
- 长期来看，在线算法接近最优固定策略

### 5.2 收敛性分析

**EMA更新的收敛**:

如果目标域是平稳的，则:
```
E[A_∞] → A*  (最优对齐矩阵)
```

收敛速度取决于α (EMA系数)

---

## 6. 实验协议

### 6.1 数据集
- BNCI2014001 (9个被试)
- LOSO交叉验证

### 6.2 超参数

**OGD方案**:
- 初始学习率: η_0 = 0.01
- 学习率衰减: η_t = η_0 / √t
- 置信度阈值: τ_conf = 0.7
- L2正则系数: λ = 0.01

**EMA方案**:
- EMA系数: α = 0.1
- 置信度阈值: τ_conf = 0.7

**Batch方案**:
- Batch大小: 32 trials
- 学习率: η = 0.01

### 6.3 评估协议

**在线评估**:
- 每个trial预测后立即评估
- 记录累积准确率曲线

**离线评估**:
- 最终LOSO准确率
- 与KSDA-static对比

---

## 7. 实现计划

### 7.1 代码结构

```
src/alignment/koopman_online.py
  - OnlineKoopmanAligner类
  - update_ogd()
  - update_ema()
  - should_update()

experiments/ksda_exp_d2.py
  - 主实验脚本
  - 在线循环实现

src/evaluation/online_analysis.py
  - 在线性能曲线
  - 对齐矩阵变化分析
```

### 7.2 实现步骤

**Week 1**:
1. 实现OnlineKoopmanAligner (3天)
2. 实现OGD和EMA更新 (2天)

**Week 2**:
1. 实现完整在线循环 (2天)
2. 运行LOSO实验 (1天)
3. 结果分析 (2天)

**Week 3** (如果需要):
1. 调试和优化 (2天)
2. 消融实验 (2天)
3. 写results memo (1天)

---

## 8. 输出文件

### 8.1 结果目录
```
results/ksda/exp_d2/<run_name>/
  - loso_results.csv
  - comparison.csv
  - summary.json
  - online_curves/
      - subject_Axx_accuracy_curve.pdf
      - subject_Axx_alignment_change.pdf
  - details/
      - subject_Axx.npz (包含A_t序列)
```

### 8.2 必须包含的字段

**details/subject_Axx.npz**:
- `y_true`: 真实标签
- `y_pred`: 预测标签
- `confidence`: 置信度序列
- `A_history`: 对齐矩阵历史 (T × m × m)
- `update_mask`: 是否更新的mask
- `alignment_change`: ||A_t - A_0||_F 序列

---

## 9. 验收标准

### 9.1 最小成功标准

1. **性能提升**:
   - KSDA-online > KSDA-static + 1%
   - 或至少在5/9个被试上优于KSDA-static

2. **适应性证据**:
   - A_t确实在变化 (||A_t - A_0|| > 0)
   - 变化与性能改善相关

3. **稳定性**:
   - 无数值不稳定
   - 性能无突降

### 9.2 理想成功标准

1. **显著提升**:
   - KSDA-online > KSDA-static + 2%
   - 超越RA baseline

2. **Session后期更好**:
   - 后50%的trial准确率 > 前50%
   - 证明适应有效

3. **最差被试改善**:
   - Subject 2等差被试性能明显提升

### 9.3 失败判定

1. **性能无提升**:
   - KSDA-online ≈ KSDA-static (差异<0.5%)

2. **性能下降**:
   - KSDA-online < KSDA-static

3. **不稳定**:
   - 性能剧烈波动
   - 数值溢出

---

## 10. 失败应对方案

### 如果性能无提升

**可能原因**:
1. 目标域与源域差异不大，不需要适应
2. 伪标签噪声太大，误导更新
3. 学习率不合适

**诊断**:
- 检查A_t是否真的在变化
- 检查伪标签准确率
- 可视化对齐前后的特征分布变化

**修复尝试**:
- 提高置信度阈值 (更保守更新)
- 调整学习率
- 改用batch更新

### 如果性能下降

**可能原因**:
1. 过度适应噪声
2. 偏离初始A_0太远
3. 正则化不足

**修复尝试**:
- 增加L2正则系数
- 降低学习率
- 只在session前期更新，后期固定

---

## 11. 成功后的下一步

### 如果达到最小成功标准

**立即行动**:
1. 写Exp-D.2 results memo
2. 分析哪些被试/哪些时段适应最有效
3. 规划Exp-D.3 (性能反馈)

### 如果达到理想成功标准

**加速推进**:
1. 立即开始Exp-D.3
2. 考虑跳过部分消融，直接进Exp-D.4
3. 开始准备论文初稿

---

## 12. 与Exp-D.3的关系

**Exp-D.2**: 在线更新A_t，但更新规则是固定的
**Exp-D.3**: 用性能反馈调整更新规则

**区别**:
- D.2: 用伪标签的置信度决定是否更新
- D.3: 用实际性能(准确率)调整更新强度

**递进关系**:
- D.2证明在线更新有效
- D.3进一步优化更新策略

---

## 13. 关键问题清单

在开始实现前，需要明确:

- [ ] 选择OGD还是EMA (推荐先OGD)
- [ ] 置信度阈值τ_conf的初始值
- [ ] 学习率η_0的初始值
- [ ] L2正则系数λ
- [ ] 是否需要正交约束
- [ ] 如何处理数值稳定性

---

**状态**: 📝 规划完成，等待Exp-D.1成功
**前置条件**: Exp-D.1 ✅
**预计开始**: Exp-D.1完成后
**预计完成**: Exp-D.1完成后2-3周
