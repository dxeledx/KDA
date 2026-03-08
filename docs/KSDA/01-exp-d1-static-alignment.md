# Exp-D.1: 静态Koopman空间对齐

**目标**: 验证在Koopman空间做对齐的基础可行性
**优先级**: 🔥 最高 (决定整个KSDA方向是否可行)
**预计时间**: 1-2周

---

## 1. 实验目的

这是KSDA的**第一个关键验证点**，要回答：

> 在Koopman线性空间做对齐，是否至少能达到协方差空间RA对齐的性能？

如果答案是否定的，后续所有工作都不必做。
如果答案是肯定的，说明这个方向有希望。

---

## 2. 方法设计

### 2.1 Pipeline

```
输入: 协方差序列 {C_t}

Step 1: Koopman Embedding
  z_t = tangent_project(C_t, M_s)  # 切空间投影 + PCA降维
  ψ_t = lift_quadratic(z_t)         # [z, z⊙z, 1]

Step 2: Static Alignment (在Koopman空间)
  ψ'_t = A @ ψ_t

  其中A是在源域学习的固定对齐矩阵

Step 3: Classification
  y_pred = LDA(ψ'_t)
```

### 2.2 对齐矩阵A的学习

**目标**: 在源域上学习A，使得对齐后的Koopman特征更有判别性

**方案A: CSP-style优化** (推荐首选)

```python
# 目标: 最大化类间距离，最小化类内距离
# 类似CSP，但在Koopman空间

for each class pair (i, j):
    # 类内协方差
    Σ_i = cov(ψ_source[y==i])
    Σ_j = cov(ψ_source[y==j])

    # 类间协方差
    Σ_between = (μ_i - μ_j)(μ_i - μ_j)^T

    # 广义特征值问题
    A = top_eigenvectors(Σ_between, Σ_i + Σ_j)
```

**方案B: 监督降维** (备选)

```python
# 用LDA的思路
A = LDA_projection_matrix(ψ_source, y_source)
```

**方案C: 最小化对齐损失** (如果A/B不行)

```python
# 直接优化分类性能
A = argmin_A Σ CE(LDA(A @ ψ_t), y_t)
```

### 2.3 关键设计决策

**决策1: Tangent空间的参考点**
- 选项A: 源域均值M_s (推荐)
- 选项B: 全局均值
- **选择A**: 与现有KCAR保持一致

**决策2: PCA降维的维度**
- 当前: r=16
- 保持不变，与现有实验一致

**决策3: Lifting方式**
- 第一版: 固定二次字典 [z, z⊙z, 1]
- 不学习lifting，保持简单

**决策4: 分类器**
- 使用LDA，与baseline保持一致
- 不引入新的分类器变量

---

## 3. 对照实验设计

### 3.1 Baseline

**Baseline-1: RA + CSP + LDA** (主要对照)
- 当前最强方法
- 性能: 43.4%

**Baseline-2: No alignment + CSP + LDA**
- 完全不对齐
- 性能: 38.0%

**Baseline-3: 协方差空间Koopman特征 + LDA**
- 用Koopman特征但不对齐
- 验证Koopman特征本身的判别性

### 3.2 KSDA变体

**KSDA-static-A**: CSP-style对齐矩阵
**KSDA-static-B**: LDA-style对齐矩阵
**KSDA-static-C**: 端到端优化对齐矩阵

---

## 4. 评估指标

### 4.1 性能指标

**主指标**:
- LOSO balanced accuracy
- 必须 ≥ RA baseline (43.4%)

**次指标**:
- Macro-F1
- Kappa
- 逐被试准确率

### 4.2 效率指标

**计算时间**:
- 训练时间 (学习A)
- 在线推理时间 (每个trial)
- 必须 < RA baseline

**内存占用**:
- 对齐矩阵大小
- 在线状态大小

### 4.3 分析指标

**对齐质量**:
- 对齐前后的类间/类内距离比
- 对齐前后的特征可视化 (t-SNE)

**Koopman特征质量**:
- 判别性分析
- 与CSP特征的对比

---

## 5. 实验协议

### 5.1 数据集
- BNCI2014001 (9个被试)
- 与stage2/stage3保持一致

### 5.2 评估协议
- LOSO交叉验证
- 每个fold:
  - 8个被试作为源域，学习A
  - 1个被试作为目标域，测试

### 5.3 预处理
- 与stage2保持完全一致
- 协方差估计、滤波等不变

### 5.4 超参数
- PCA维度: r=16
- Lifting: 二次字典
- LDA: 默认参数
- 对齐矩阵A: 保留top-k个特征向量 (k待定)

---

## 6. 实现计划

### 6.1 代码结构

```
src/alignment/koopman_alignment.py
  - KoopmanAligner类
  - learn_alignment_matrix()
  - apply_alignment()

experiments/ksda_exp_d1.py
  - 主实验脚本
  - 运行所有baseline和KSDA变体

src/evaluation/ksda_analysis.py
  - 对齐质量分析
  - 特征可视化
```

### 6.2 复用现有代码

**可以直接复用**:
- `src/evaluation/kcar_analysis.py` 中的:
  - `TangentProjector`
  - `_lift_quadratic()`
  - `fit_tangent_projector()`

**需要新增**:
- 对齐矩阵学习
- Koopman空间分类pipeline

### 6.3 实现步骤

**Week 1**:
1. 实现KoopmanAligner类 (2天)
2. 实现CSP-style对齐矩阵学习 (2天)
3. 单元测试 (1天)

**Week 2**:
1. 实现完整实验脚本 (2天)
2. 运行LOSO实验 (1天)
3. 结果分析和可视化 (2天)

---

## 7. 输出文件

### 7.1 结果目录
```
results/ksda/exp_d1/<run_name>/
  - loso_results.csv
  - comparison.csv
  - summary.json
  - timing.json
  - details/
      - subject_Axx.npz
  - figures/
      - accuracy_comparison.pdf
      - feature_tsne_before_after.pdf
      - alignment_quality.pdf
```

### 7.2 必须包含的字段

**loso_results.csv**:
- target_subject
- method (baseline-ra, ksda-static-a, etc.)
- accuracy, kappa, f1_macro
- train_time, test_time_per_trial

**comparison.csv**:
- method
- accuracy_mean, accuracy_std
- vs_ra_delta
- train_time_mean, test_time_mean

**summary.json**:
```json
{
  "ksda_static_a": {
    "accuracy_mean": ...,
    "vs_ra_delta": ...,
    "wins_vs_ra": ...,
    "speedup": ...
  },
  ...
}
```

---

## 8. 验收标准

### 8.1 最小成功标准 (必须满足)

1. **性能不低于RA**:
   - KSDA-static accuracy ≥ 43.4%
   - 或至少 ≥ 42.0% (允许1.4%的小幅下降)

2. **计算效率更高**:
   - 在线推理时间 < RA
   - 至少快2倍

3. **实现正确性**:
   - 通过单元测试
   - 对齐矩阵A是有限值
   - 对齐后特征维度正确

### 8.2 理想成功标准 (期望达到)

1. **性能优于RA**:
   - KSDA-static > 44.0%
   - 至少在5/9个被试上优于RA

2. **显著加速**:
   - 快5-10倍

3. **对齐质量提升**:
   - 类间/类内距离比 > RA
   - t-SNE可视化显示更好的聚类

### 8.3 失败判定

如果满足以下任一条件，判定失败:

1. **性能显著下降**:
   - KSDA-static < 40.0%
   - 比RA低3%以上

2. **计算效率无优势**:
   - 在线推理时间 ≥ RA

3. **实现有bug**:
   - 对齐后性能比对齐前更差
   - 数值不稳定

---

## 9. 失败应对方案

### 如果性能不达标

**诊断步骤**:
1. 检查Koopman特征本身的判别性
   - 对比 ψ_t 直接分类 vs CSP特征分类
   - 如果Koopman特征本身就差，问题在lifting

2. 检查对齐矩阵A的质量
   - 可视化A的特征向量
   - 检查是否过拟合或欠拟合

3. 检查对齐前后的特征分布
   - t-SNE可视化
   - 类间/类内距离统计

**可能的修复**:
- 改进lifting (更高阶多项式，或学习MLP)
- 改进对齐矩阵学习方法
- 调整PCA维度

**止损规则**:
- 如果尝试3种对齐矩阵学习方法都失败
- 且Koopman特征本身判别性就差
- 则放弃KSDA方向，回到窗口级KCAR

---

## 10. 成功后的下一步

### 如果达到最小成功标准

**立即行动**:
1. 写Exp-D.1 results memo
2. 分析成功的关键因素
3. 规划Exp-D.2 (在线更新)

### 如果达到理想成功标准

**加速推进**:
1. 立即开始Exp-D.2和D.3并行
2. 考虑直接跳到Exp-D.4 (风险感知)
3. 开始准备论文Method section

---

## 11. 时间线

**Day 1-2**: 实现KoopmanAligner
**Day 3-4**: 实现对齐矩阵学习
**Day 5**: 单元测试和调试
**Day 6-7**: 运行LOSO实验
**Day 8-9**: 结果分析和可视化
**Day 10**: 写results memo和决策

**总计**: 10个工作日 (2周)

---

## 12. 关键问题清单

在开始实现前，需要明确:

- [ ] 对齐矩阵A的学习方法 (CSP-style vs LDA-style vs 端到端)
- [ ] 对齐矩阵A的维度 (保留多少个特征向量)
- [ ] 是否需要正则化 (L2 penalty on A)
- [ ] 如何处理数值稳定性 (矩阵求逆，特征值分解)
- [ ] 是否需要归一化对齐后的特征

---

## 13. 与现有工作的关系

**复用**:
- Stage3的Koopman embedding代码
- Stage2的评估框架

**新增**:
- Koopman空间对齐
- 对齐质量分析

**不做**:
- 不做动态更新 (留给Exp-D.2)
- 不做性能反馈 (留给Exp-D.3)
- 不做KCAR集成 (留给Exp-D.4)

---

**状态**: 📝 规划完成，等待实现
**负责人**: TBD
**预计开始**: 2026-03-08
**预计完成**: 2026-03-22
