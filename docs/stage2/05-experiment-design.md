# 阶段二 - 实验设计 (Experiment Design)

## 1. 实验目标

### 1.1 核心研究问题

**RQ1**: 条件对齐是否比固定对齐更有效？
- 对比：固定权重 (w=0, 0.5, 1) vs 动态权重 (w_t = g(c_t))

**RQ2**: 行为引导反馈是否提升性能？
- 对比：无反馈 vs 有反馈

**RQ3**: 哪些上下文特征最重要？
- 消融研究：d_src, d_tgt, σ_recent, H_t, conf_avg, KL_div

**RQ4**: DCA-BGF是否改善表征-行为一致性？
- 指标：表征-行为相关系数 r

---

## 2. 实验协议

### 2.1 数据集

**主数据集**: BCI Competition IV Dataset 2a
- 9个被试
- 4类运动想象（左手、右手、双脚、舌头）
- 每个被试：288个训练trial，288个测试trial
- 采样率：250 Hz
- 通道数：22个EEG通道

**预处理**：
- 带通滤波：8-30 Hz
- Epoch提取：0.5-2.5s（相对于cue onset）
- 基线校正：使用cue前0.5s

### 2.2 评估协议

**协议1: Leave-One-Subject-Out (LOSO)**
```python
for target_subject in range(1, 10):
    # 源域：其他8个被试的训练数据
    source_subjects = [s for s in range(1, 10) if s != target_subject]
    X_source = concatenate([load_subject(s, 'train') for s in source_subjects])

    # 目标域：当前被试的测试数据
    X_target = load_subject(target_subject, 'test')

    # 训练和评估
    model.fit(X_source)
    y_pred = model.predict(X_target)
    acc = accuracy_score(y_target, y_pred)
```

**协议2: Within-Subject (验证baseline)**
```python
# 用于验证实现是否正确
X_train, y_train = load_subject(subject_id, 'train')
X_test, y_test = load_subject(subject_id, 'test')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
```

**预期**：Within-subject准确率应该 > 70%

### 2.3 对比方法

| 方法 | 描述 | 来源 |
|------|------|------|
| **CSP+LDA** | 无对齐的baseline | 阶段一 |
| **EA+CSP+LDA** | Euclidean Alignment | 阶段一 |
| **RA+CSP+LDA** | Riemannian Alignment | 阶段一 |
| **CORAL** | Correlation Alignment | 论文有源码 |
| **OTTA** | Online Test-Time Adaptation | arXiv:2311.18520 |
| **DCA-BGF (Ours)** | 完整方法 | 本研究 |

**简化版本**（如果时间紧张）：
- 只对比：CSP+LDA, RA+CSP+LDA, DCA-BGF

---

## 3. 消融研究

### 3.1 实验1：条件对齐的有效性

**目标**：验证动态对齐权重比固定权重更好

**对比方法**：
```python
# Baseline 1: 不对齐
w_t = 0.0

# Baseline 2: 部分对齐
w_t = 0.5

# Baseline 3: 完全对齐
w_t = 1.0

# Ablation 1: 只用几何距离
c_t = [d_src]
w_t = g(c_t)

# Ablation 2: 几何距离 + 方差
c_t = [d_src, d_tgt, σ_recent]
w_t = g(c_t)

# 完整版本: 所有上下文特征
c_t = [d_src, d_tgt, σ_recent, H_t, conf_avg, KL_div]
w_t = g(c_t)
```

**评估指标**：
- 跨被试准确率（LOSO）
- 对齐权重的方差（衡量动态性）

**预期结果**：
- 动态权重 > 所有固定权重
- 完整上下文 > 简化上下文

**关键图表**：
```python
# 图1：不同方法的准确率对比（箱线图）
plt.boxplot([acc_w0, acc_w05, acc_w1, acc_dynamic])
plt.xticks([1, 2, 3, 4], ['w=0', 'w=0.5', 'w=1', 'Dynamic'])
plt.ylabel('Accuracy')
plt.title('Comparison of Fixed vs Dynamic Alignment')
```

### 3.2 实验2：行为引导反馈的有效性

**目标**：验证反馈机制提升性能

**对比方法**：
```python
# Baseline: 无反馈
dca_bgf = DCABGF(use_feedback=False)

# Ablation 1: 只有不确定性反馈
feedback = BehaviorGuidedFeedback(use_uncertainty=True, use_trend=False)

# Ablation 2: 只有趋势反馈
feedback = BehaviorGuidedFeedback(use_uncertainty=False, use_trend=True)

# 完整版本: 所有反馈规则
feedback = BehaviorGuidedFeedback(use_uncertainty=True, use_trend=True)
```

**评估指标**：
- 跨被试准确率
- Within-session稳定性（准确率方差）
- 反馈触发次数

**预期结果**：
- 有反馈 > 无反馈
- 在session后期（trial 100-288）性能更稳定

**关键图表**：
```python
# 图2：准确率随时间变化（滑动窗口）
window_size = 20
for method in ['No Feedback', 'With Feedback']:
    acc_window = compute_sliding_accuracy(y_true, y_pred, window_size)
    plt.plot(acc_window, label=method)
plt.xlabel('Trial')
plt.ylabel('Accuracy (sliding window)')
plt.legend()
plt.title('Within-Session Performance Stability')
```

### 3.3 实验3：上下文特征的贡献

**目标**：识别最重要的上下文特征

**对比方法**：
```python
# 逐个添加特征
contexts = [
    [d_src],                              # 只用源域距离
    [d_src, d_tgt],                       # + 目标域距离
    [d_src, d_tgt, σ_recent],             # + 方差
    [d_src, d_tgt, σ_recent, H_t],        # + 熵
    [d_src, d_tgt, σ_recent, H_t, conf_avg],  # + 置信度
    [d_src, d_tgt, σ_recent, H_t, conf_avg, KL_div]  # 完整
]
```

**评估指标**：
- 准确率提升（相对于前一个版本）
- 特征重要性（如果用线性模型）

**预期结果**：
- d_src是最重要的特征
- 添加行为特征（H_t, conf_avg）有额外提升

**关键图表**：
```python
# 图3：特征重要性（条形图）
feature_names = ['d_src', 'd_tgt', 'σ_recent', 'H_t', 'conf_avg', 'KL_div']
importances = compute_feature_importance(model)
plt.bar(feature_names, importances)
plt.ylabel('Importance')
plt.title('Context Feature Importance')
```

### 3.4 实验4：表征-行为一致性

**目标**：验证DCA-BGF改善表征-行为一致性

**评估方法**：
```python
# 对每对被试(i, j)
for i in range(1, 10):
    for j in range(i+1, 10):
        # 1. 计算表征相似度
        features_i = extract_features(subject_i, method)
        features_j = extract_features(subject_j, method)
        rep_sim = cka(features_i, features_j)

        # 2. 计算行为一致性
        acc_i = evaluate(subject_i, method)
        acc_j = evaluate(subject_j, method)
        beh_consistency = 1 - abs(acc_i - acc_j)

        # 3. 记录
        rep_sims.append(rep_sim)
        beh_consistencies.append(beh_consistency)

# 4. 计算相关系数
r = pearsonr(rep_sims, beh_consistencies)
```

**对比方法**：
- EA: r ≈ 0.64（阶段一结果）
- RA: r ≈ 0.64（阶段一结果）
- DCA-BGF: r > 0.75（目标）

**关键图表**：
```python
# 图4：表征-行为散点图
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(rep_sims_ea, beh_consistencies_ea)
plt.xlabel('Representation Similarity (CKA)')
plt.ylabel('Behavior Consistency')
plt.title(f'EA (r={r_ea:.2f})')

plt.subplot(1, 3, 2)
plt.scatter(rep_sims_ra, beh_consistencies_ra)
plt.xlabel('Representation Similarity (CKA)')
plt.ylabel('Behavior Consistency')
plt.title(f'RA (r={r_ra:.2f})')

plt.subplot(1, 3, 3)
plt.scatter(rep_sims_dca, beh_consistencies_dca)
plt.xlabel('Representation Similarity (CKA)')
plt.ylabel('Behavior Consistency')
plt.title(f'DCA-BGF (r={r_dca:.2f})')
```

---

## 4. 评估指标

### 4.1 主要指标

**指标1：跨被试准确率**
```python
acc_loso = mean([accuracy_score(y_true_i, y_pred_i) for i in subjects])
```

**目标**：
- Baseline (RA): 60-70%
- DCA-BGF: >65-75% (+5%)

**指标2：表征-行为相关系数**
```python
r = pearsonr(rep_similarities, beh_consistencies)[0]
```

**目标**：
- Baseline (RA): r ≈ 0.64
- DCA-BGF: r > 0.75

### 4.2 辅助指标

**指标3：Within-session稳定性**
```python
# 计算每个被试在session后期的准确率方差
stability = []
for subject in subjects:
    acc_late = compute_accuracy(y_true[100:], y_pred[100:])
    stability.append(np.var(acc_late))
stability_score = 1 / mean(stability)  # 方差越小越好
```

**指标4：计算效率**
```python
import time
start = time.time()
y_pred = model.predict(X_test)
inference_time = time.time() - start
time_per_trial = inference_time / len(X_test)
```

**目标**：< 10ms/trial

**指标5：对齐权重动态性**
```python
# 衡量对齐权重的变化程度
w_variance = np.var(w_t_values)
w_range = np.max(w_t_values) - np.min(w_t_values)
```

**预期**：
- w_variance > 0.01（有动态变化）
- w_range > 0.3（使用了较大的权重范围）

### 4.3 统计显著性检验

**方法1：配对t检验**
```python
from scipy.stats import ttest_rel

# 对比DCA-BGF和RA在9个被试上的准确率
acc_dca = [acc_dca_subject_i for i in range(1, 10)]
acc_ra = [acc_ra_subject_i for i in range(1, 10)]

t_stat, p_value = ttest_rel(acc_dca, acc_ra)
print(f"t={t_stat:.3f}, p={p_value:.4f}")
```

**显著性水平**：α = 0.05

**方法2：Wilcoxon符号秩检验（非参数）**
```python
from scipy.stats import wilcoxon

w_stat, p_value = wilcoxon(acc_dca, acc_ra)
```

**报告格式**：
```
DCA-BGF achieved significantly higher accuracy than RA
(68.5% vs 62.3%, p < 0.01, paired t-test).
```

---

## 5. 实验流程

### 5.1 完整实验流程

```python
# experiments/full_experiment.py

def run_full_experiment():
    """运行完整实验"""
    results = {
        'methods': {},
        'ablations': {},
        'rep_beh': {}
    }

    # 1. 对比方法
    methods = {
        'CSP+LDA': CSP_LDA(),
        'EA': EA_CSP_LDA(),
        'RA': RA_CSP_LDA(),
        'DCA-BGF': DCABGF()
    }

    for method_name, method in methods.items():
        print(f"\n=== Running {method_name} ===")
        acc_list = []

        for target_subject in range(1, 10):
            # LOSO
            acc = run_loso(method, target_subject)
            acc_list.append(acc)
            print(f"Subject {target_subject}: {acc:.4f}")

        results['methods'][method_name] = {
            'acc_mean': np.mean(acc_list),
            'acc_std': np.std(acc_list),
            'acc_list': acc_list
        }

    # 2. 消融研究
    ablations = {
        'w=0': DCABGF(w_fixed=0.0),
        'w=0.5': DCABGF(w_fixed=0.5),
        'w=1.0': DCABGF(w_fixed=1.0),
        'Dynamic (no feedback)': DCABGF(use_feedback=False),
        'Dynamic (with feedback)': DCABGF(use_feedback=True)
    }

    for ablation_name, method in ablations.items():
        print(f"\n=== Running {ablation_name} ===")
        acc_list = []

        for target_subject in range(1, 10):
            acc = run_loso(method, target_subject)
            acc_list.append(acc)

        results['ablations'][ablation_name] = {
            'acc_mean': np.mean(acc_list),
            'acc_std': np.std(acc_list),
            'acc_list': acc_list
        }

    # 3. 表征-行为分析
    for method_name in ['EA', 'RA', 'DCA-BGF']:
        rep_sims, beh_cons = compute_rep_beh_consistency(methods[method_name])
        r = pearsonr(rep_sims, beh_cons)[0]
        results['rep_beh'][method_name] = {
            'r': r,
            'rep_sims': rep_sims,
            'beh_cons': beh_cons
        }

    # 4. 保存结果
    save_results(results, 'results/full_experiment.pkl')

    # 5. 生成报告
    generate_report(results)

    return results
```

### 5.2 时间安排

**Week 1-2: MVP实验**
- Day 1-3: 实现MVP版本
- Day 4-5: 调试和验证
- Day 6-7: 在3个被试上测试

**Week 3-4: 完整实验**
- Day 1-2: 实现完整版本
- Day 3-5: 运行LOSO实验（9个被试）
- Day 6-7: 消融研究

**Week 5: 分析和可视化**
- Day 1-2: 表征-行为分析
- Day 3-4: 生成所有图表
- Day 5: 统计检验

**Week 6: 完善和撰写**
- Day 1-3: 补充实验（如果需要）
- Day 4-7: 撰写方法和结果部分

---

## 6. 结果报告

### 6.1 表格格式

**表1：主要结果对比**
```
| Method      | Accuracy (%) | Std (%) | p-value |
|-------------|--------------|---------|---------|
| CSP+LDA     | 45.2         | 8.3     | -       |
| EA          | 58.7         | 7.1     | <0.001  |
| RA          | 62.3         | 6.5     | <0.001  |
| DCA-BGF     | 68.5         | 5.8     | <0.01   |
```

**表2：消融研究**
```
| Configuration           | Accuracy (%) | Δ vs Baseline |
|------------------------|--------------|---------------|
| w=0 (No alignment)     | 45.2         | -             |
| w=0.5 (Fixed)          | 60.1         | +14.9         |
| w=1.0 (Full alignment) | 62.3         | +17.1         |
| Dynamic (no feedback)  | 65.7         | +20.5         |
| Dynamic (with feedback)| 68.5         | +23.3         |
```

**表3：表征-行为一致性**
```
| Method   | Correlation (r) | p-value |
|----------|----------------|---------|
| EA       | 0.64           | <0.001  |
| RA       | 0.64           | <0.001  |
| DCA-BGF  | 0.78           | <0.001  |
```

### 6.2 图表清单

**必须的图表**：
1. 准确率对比（箱线图）
2. 表征-行为散点图（3个子图：EA, RA, DCA-BGF）
3. 对齐权重演化（线图）
4. Within-session稳定性（线图）

**可选的图表**：
5. 特征重要性（条形图）
6. 反馈触发时刻（散点图）
7. 跨被试性能热图
8. 混淆矩阵

---

## 7. 失败预案

### 7.1 如果DCA-BGF效果不如RA怎么办？

**诊断步骤**：
1. 检查实现是否正确（单元测试）
2. 可视化对齐权重（是否有动态变化）
3. 检查上下文特征（是否有区分度）
4. 尝试不同的超参数

**Plan B**：
- 降级到更简单的版本（只用d_src）
- 尝试不同的基础对齐方法（RA而不是EA）
- 增加数据增强

### 7.2 如果表征-行为相关性没有提升怎么办？

**诊断步骤**：
1. 检查CKA计算是否正确
2. 尝试不同的表征相似度指标（如CCA）
3. 分析失败案例（哪些被试对表征相似但性能差异大）

**Plan B**：
- 弱化这个贡献点，强调准确率提升
- 改投应用型期刊（不强调理论创新）

---

## 8. 下一步

完成实验设计后，继续阅读：
- **06-expected-results.md**：了解预期结果和风险对冲

---

**文档版本**: v1.0
**最后更新**: 2026-03-05
**状态**: 实验设计完成
