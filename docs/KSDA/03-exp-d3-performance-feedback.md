# Exp-D.3: 性能驱动的Koopman算子修正

**前置条件**: Exp-D.2成功 (在线更新A_t有效)
**目标**: 用在线性能反馈修正Koopman算子K_t，使其更有利于分类
**优先级**: 🔥 高
**预计时间**: 2-3周

---

## 1. 实验目的

在Exp-D.2的基础上，回答：

> 用在线分类性能作为反馈，修正Koopman算子K_t的估计，能否进一步提升性能？

**核心假设**:
- 当前K_t用无监督方式拟合（最小化预测误差）
- 但预测误差 ≠ 分类性能
- 用性能反馈可以让K_t学到更有判别性的动力学

**与Exp-D.2的区别**:
- D.2: 在线更新对齐矩阵A_t
- D.3: 在线修正Koopman算子K_t
- D.3是更深层的适应

---

## 2. 方法设计

### 2.1 Pipeline

```
初始化:
  K_s = fit_global_koopman(source_data)  # 源域算子
  K_t = K_s.copy()                       # 初始目标域算子
  A_0 = learn_alignment_matrix(source_data)

在线循环 (t = 1, 2, ..., T):
  Step 1: Koopman Embedding
    z_t = tangent_project(C_t, M_s)
    ψ_t = lift_quadratic(z_t)

  Step 2: Dynamic Alignment
    ψ'_t = A_t @ ψ_t

  Step 3: Prediction
    y_pred_t = LDA(ψ'_t)
    p_t = softmax(LDA.decision_function(ψ'_t))

  Step 4: Performance Feedback (关键创新)
    acc_recent = compute_recent_accuracy(window=32)

    # 根据性能调整K_t更新策略
    if acc_recent > threshold_high:
      # 性能好，当前动力学可靠，正常更新
      K_t = update_koopman(K_t, ψ_t, weight=1.0)
    elif acc_recent < threshold_low:
      # 性能差，当前动力学不可靠，保守更新或不更新
      K_t = update_koopman(K_t, ψ_t, weight=0.1)
    else:
      # 中等性能，适度更新
      K_t = update_koopman(K_t, ψ_t, weight=0.5)

  Step 5: Update Alignment (from Exp-D.2)
    A_{t+1} = update_alignment(A_t, ψ_t, feedback_t)
```

### 2.2 性能加权的Koopman更新

**方案A: 性能加权最小二乘** (推荐首选)

```python
# 标准Koopman拟合
K = argmin Σ ||ψ_{t+1} - K @ ψ_t||²

# 性能加权版本
K = argmin Σ w_perf(t) * ||ψ_{t+1} - K @ ψ_t||²

# 其中 w_perf(t) 根据最近性能动态调整:
w_perf(t) = {
    1.0,  if acc_recent > 0.7  # 性能好，高权重
    0.5,  if 0.5 < acc_recent ≤ 0.7  # 中等性能
    0.1,  if acc_recent ≤ 0.5  # 性能差，低权重
}
```

**方案B: 判别式Koopman学习** (更激进)

```python
# 联合目标: 动力学 + 分类
L = L_dynamics + λ * L_classification

# 其中:
L_dynamics = Σ ||ψ_{t+1} - K @ ψ_t||²
L_classification = Σ CE(LDA(K @ ψ_t), y_pseudo_t)

# 在线梯度更新
K_{t+1} = K_t - η * (∇_K L_dynamics + λ * ∇_K L_classification)
```

**方案C: 强化学习框架** (最激进)

```python
# 把K_t更新看作强化学习问题
# State: 当前Koopman特征 ψ_t
# Action: 更新K_t的方向
# Reward: 在线分类准确率

# 策略梯度更新
K_{t+1} = K_t + η * ∇_K [R_t * log P(action|ψ_t, K_t)]
```

### 2.3 关键设计决策

**决策1: 性能窗口大小**
- 选项A: 固定窗口 (如32个trial)
- 选项B: 自适应窗口 (性能稳定时扩大)
- **选择A**: 简单稳定

**决策2: 性能阈值**
- threshold_high = 0.7
- threshold_low = 0.5
- 可以根据baseline性能调整

**决策3: K_t更新频率**
- 选项A: 每个trial都更新
- 选项B: 每N个trial更新一次
- **选择B**: 更稳定，N=8或16

**决策4: 是否同时更新A_t和K_t**
- 选项A: 同时更新 (可能不稳定)
- 选项B: 交替更新 (更稳定)
- **选择B**: 先更新K_t，再更新A_t

---

## 3. 对照实验设计

### 3.1 Baseline

**Baseline-1: KSDA-online** (主要对照)
- 在线更新A_t，但K_t固定
- 来自Exp-D.2

**Baseline-2: KSDA-static**
- A_t和K_t都固定
- 来自Exp-D.1

### 3.2 KSDA-perf变体

**KSDA-perf-A**: 性能加权最小二乘
**KSDA-perf-B**: 判别式Koopman学习
**KSDA-perf-C**: 强化学习框架

**消融变体**:
- **KSDA-perf-no-weight**: 不用性能加权，均匀更新
- **KSDA-perf-fixed-A**: 只更新K_t，A_t固定
- **KSDA-perf-alternate**: 交替更新K_t和A_t

---

## 4. 评估指标

### 4.1 性能指标

**主指标**:
- LOSO balanced accuracy
- 必须 > KSDA-online

**次指标**:
- Session后期性能提升幅度
- 最差被试改善幅度

### 4.2 Koopman质量指标

**动力学预测误差**:
- ||ψ_{t+1} - K_t @ ψ_t||² 随时间变化
- 性能好的时段 vs 性能差的时段

**判别性分析**:
- 用K_t变换后的特征分类准确率
- 对比原始ψ_t的分类准确率

### 4.3 适应性指标

**K_t变化**:
- ||K_t - K_0||_F 随时间变化
- K_t的谱变化

**性能-动力学相关性**:
- 性能提升 vs K_t变化的相关性

---

## 5. 理论分析

### 5.1 为什么性能反馈有效？

**直觉**:
- 无监督Koopman拟合学到的是"数据的动力学"
- 但不一定是"有利于分类的动力学"
- 性能反馈引导K_t学习判别性动力学

**类比**:
- 无监督: 学习"物体如何运动"
- 有监督: 学习"如何区分不同类别的运动"

### 5.2 收敛性分析

**假设**:
- 存在最优K*使得分类性能最大化
- 性能反馈提供了梯度信息

**结论**:
- 在适当的学习率下，K_t → K*

---

## 6. 实验协议

### 6.1 数据集
- BNCI2014001 (9个被试)
- LOSO交叉验证

### 6.2 超参数

**性能加权方案**:
- 性能窗口: 32 trials
- threshold_high: 0.7
- threshold_low: 0.5
- K_t更新频率: 每8个trial

**判别式学习方案**:
- λ (分类损失权重): 0.1
- 学习率: η = 0.001
- 更新频率: 每8个trial

### 6.3 评估协议

**在线评估**:
- 记录每个trial的性能
- 记录K_t的变化

**离线评估**:
- 最终LOSO准确率
- 与KSDA-online对比

---

## 7. 实现计划

### 7.1 代码结构

```
src/alignment/koopman_performance.py
  - PerformanceWeightedKoopman类
  - compute_recent_accuracy()
  - update_koopman_with_performance()

experiments/ksda_exp_d3.py
  - 主实验脚本
  - 性能反馈循环

src/evaluation/koopman_quality.py
  - 动力学预测误差分析
  - 判别性分析
```

### 7.2 实现步骤

**Week 1**:
1. 实现性能加权Koopman更新 (3天)
2. 实现性能窗口计算 (1天)
3. 单元测试 (1天)

**Week 2**:
1. 实现完整在线循环 (2天)
2. 运行LOSO实验 (1天)
3. 结果分析 (2天)

**Week 3** (如果需要):
1. 实现判别式学习方案 (2天)
2. 对比实验 (2天)
3. 写results memo (1天)

---

## 8. 输出文件

### 8.1 结果目录
```
results/ksda/exp_d3/<run_name>/
  - loso_results.csv
  - comparison.csv
  - summary.json
  - koopman_quality/
      - subject_Axx_prediction_error.pdf
      - subject_Axx_koopman_change.pdf
  - details/
      - subject_Axx.npz (包含K_t序列)
```

### 8.2 必须包含的字段

**details/subject_Axx.npz**:
- `K_history`: Koopman算子历史
- `prediction_error`: 动力学预测误差序列
- `performance_window`: 滑窗性能序列
- `update_weight`: 性能加权序列

---

## 9. 验收标准

### 9.1 最小成功标准

1. **性能提升**:
   - KSDA-perf > KSDA-online + 1%

2. **动力学改善**:
   - 性能好的时段，预测误差更低
   - 证明K_t确实在学习更好的动力学

3. **稳定性**:
   - K_t更新稳定，无数值问题

### 9.2 理想成功标准

1. **显著提升**:
   - KSDA-perf > KSDA-online + 2%
   - 超越RA baseline

2. **判别性增强**:
   - K_t变换后的特征分类准确率 > 原始特征

3. **最差被试改善**:
   - Subject 2等差被试性能明显提升

### 9.3 失败判定

1. **性能无提升**:
   - KSDA-perf ≈ KSDA-online

2. **性能下降**:
   - KSDA-perf < KSDA-online

3. **不稳定**:
   - K_t更新导致性能剧烈波动

---

## 10. 失败应对方案

### 如果性能无提升

**可能原因**:
1. 性能反馈噪声太大
2. K_t更新频率不合适
3. 性能窗口太小或太大

**修复尝试**:
- 增大性能窗口
- 降低K_t更新频率
- 使用更平滑的性能估计

### 如果性能下降

**可能原因**:
1. 过度适应噪声
2. K_t和A_t更新冲突
3. 判别式目标破坏了动力学结构

**修复尝试**:
- 交替更新K_t和A_t
- 降低判别式损失权重λ
- 只在高置信度时更新K_t

---

## 11. 成功后的下一步

### 如果达到最小成功标准

**立即行动**:
1. 写Exp-D.3 results memo
2. 分析性能反馈的作用机制
3. 规划Exp-D.4 (风险感知)

### 如果达到理想成功标准

**加速推进**:
1. 立即开始Exp-D.4
2. 开始准备论文Method section
3. 考虑多数据集验证

---

## 12. 与Exp-D.4的关系

**Exp-D.3**: 用性能反馈修正K_t
**Exp-D.4**: 用KCAR控制对齐强度

**递进关系**:
- D.3: 让K_t更有判别性
- D.4: 用K_t计算的KCAR控制对齐
- 两者结合: 完整的KSDA框架

---

## 13. 关键问题清单

在开始实现前，需要明确:

- [ ] 选择性能加权还是判别式学习 (推荐先性能加权)
- [ ] 性能窗口大小
- [ ] 性能阈值 (high/low)
- [ ] K_t更新频率
- [ ] 是否同时更新A_t和K_t
- [ ] 如何处理数值稳定性

---

**状态**: 📝 规划完成，等待Exp-D.2成功
**前置条件**: Exp-D.2 ✅
**预计开始**: Exp-D.2完成后
**预计完成**: Exp-D.2完成后2-3周
