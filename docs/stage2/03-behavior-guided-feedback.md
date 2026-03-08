# 阶段二 - 行为引导反馈 (Behavior-Guided Feedback)

## 1. 核心设计理念

### 1.1 为什么需要行为反馈？

**阶段一的关键发现**：
- ✅ 表征相似度提升（CKA: 0.39 → 0.64）
- ⚠️ 但性能提升有限（Acc: 38% → 42%）
- ❌ **表征-行为不一致**：表征对齐 ≠ 性能提升

**传统方法的盲点**：
- EA/RA只关注表征对齐，忽略了最终的行为性能
- 没有闭环反馈机制，无法根据在线性能调整策略
- 假设表征对齐自动带来性能提升（但阶段一证明这不成立）

**行为引导反馈的价值**：
- **闭环优化**：利用在线性能指标调整对齐策略
- **表征-行为桥接**：显式建模表征相似度与行为性能的关系
- **持续适应**：在session过程中持续自我修正

### 1.2 设计原则

1. **无需标签**：只使用预测置信度和几何信息（无监督）
2. **轻量级**：计算开销小，适合在线场景
3. **稳定性**：避免反馈震荡，使用滑动窗口平滑
4. **可解释性**：反馈规则应该有明确的物理意义

---

## 2. 监控指标设计

### 2.1 性能监控指标

#### 指标1：滑动窗口准确率（伪标签）

**定义**：
```python
acc_window = (1/W) * Σ_{i=t-W+1}^{t} 𝟙(ŷ_i = y_i)
```

**问题**：在线阶段没有真实标签 y_i！

**解决方案**：使用高置信度预测作为伪标签
```python
# 只统计置信度 > threshold 的trial
pseudo_labels = [ŷ_i for i in range(t-W, t) if conf_i > 0.8]
acc_window = accuracy(pseudo_labels, true_labels)  # 假设有部分标签
```

**更实用的版本**：不依赖任何标签
```python
# 使用置信度作为性能代理
acc_proxy = mean([max(p_k^{(i)}) for i in range(t-W, t)])
```

**直觉**：
- 置信度高 → 模型确定 → 性能可能好
- 置信度低 → 模型不确定 → 性能可能差

**参数**：
- 窗口大小 W = 10（最近10个trial）
- 置信度阈值 = 0.8

#### 指标2：预测熵

**定义**：
```python
H_t = -Σ_k p_k log p_k
```

其中 p_k 是分类器输出的类别概率

**物理意义**：
- H_t = 0：完全确定（p = [1, 0, 0, 0]）
- H_t = log(K)：完全不确定（p = [0.25, 0.25, 0.25, 0.25]）

**滑动窗口平均**：
```python
H_avg = mean([H_i for i in range(t-W, t)])
```

**直觉**：
- H_avg 高 → 模型不确定 → 需要调整对齐
- H_avg 低 → 模型确定 → 当前策略有效

#### 指标3：置信度趋势

**定义**：
```python
conf_trend = (conf_t - conf_{t-W}) / W
```

**物理意义**：
- conf_trend > 0：置信度上升 → 性能改善
- conf_trend < 0：置信度下降 → 性能退化

**更稳健的版本**：使用线性回归斜率
```python
from scipy.stats import linregress
slope, _, _, _, _ = linregress(range(W), conf_history[-W:])
conf_trend = slope
```

#### 指标4：类别分布偏差

**定义**：
```python
class_dist = [count(ŷ=k) / W for k in classes]
KL_div = KL(class_dist || uniform_dist)
```

**物理意义**：
- KL_div = 0：类别均匀分布（理想情况）
- KL_div 大：类别不平衡 → 可能有偏差

**直觉**：
- 如果模型总是预测同一个类别 → 说明对齐有问题
- 需要调整对齐策略或重新校准

### 2.2 表征监控指标

#### 指标5：特征漂移

**定义**：
```python
drift_t = ||μ_tgt^{(t)} - μ_tgt^{(t-W)}||
```

其中 μ_tgt^{(t)} 是当前时刻的目标域中心

**物理意义**：
- drift_t 大 → 分布在快速变化 → 需要更频繁的对齐
- drift_t 小 → 分布稳定 → 可以减少对齐

#### 指标6：对齐质量

**定义**：
```python
alignment_quality = 1 - ||μ_tgt - μ_src|| / ||μ_tgt^{(0)} - μ_src||
```

**物理意义**：
- alignment_quality = 1：完全对齐
- alignment_quality = 0：未对齐

**直觉**：
- 如果对齐质量下降 → 需要增强对齐
- 如果对齐质量过高 → 可能过度对齐，损失个体特征

---

## 3. 反馈规则设计

### 3.1 MVP版本：简单规则

**规则1：不确定性反馈**
```python
if H_avg > H_threshold_high:
    # 模型很不确定 → 增强对齐
    w_t = min(w_t + α, 1.0)
elif H_avg < H_threshold_low:
    # 模型很确定 → 减弱对齐（保留个体特征）
    w_t = max(w_t - β, 0.0)
else:
    # 保持当前对齐强度
    w_t = w_t
```

**参数设置**：
- `H_threshold_high = 0.8 * log(K)`（K是类别数）
- `H_threshold_low = 0.3 * log(K)`
- `α = 0.1`（增加步长）
- `β = 0.05`（减少步长，更保守）

**直觉**：
- 不确定时增强对齐（向源域靠拢）
- 确定时减弱对齐（保留个体特征）
- 增加比减少更激进（α > β）

**规则2：置信度趋势反馈**
```python
if conf_trend < -0.05:
    # 置信度持续下降 → 性能退化
    # 重新估计目标域中心
    μ_tgt = mean([x_i for i in range(t-W, t)])
    # 增强对齐
    w_t = min(w_t + 0.15, 1.0)
```

**直觉**：
- 性能退化可能是因为分布漂移
- 重新估计目标域中心，增强对齐

**规则3：类别偏差反馈**
```python
if KL_div > KL_threshold:
    # 类别分布不均匀 → 可能有偏差
    # 使用高置信度样本的伪标签微调分类器
    high_conf_samples = [(x_i, ŷ_i) for i in range(t-W, t) if conf_i > 0.9]
    classifier.partial_fit(high_conf_samples)
```

**参数设置**：
- `KL_threshold = 0.5`

**直觉**：
- 类别不平衡可能是分类器偏差
- 用高置信度样本校准分类器

### 3.2 完整版本：自适应反馈

**核心思想**：根据多个指标综合决策

```python
def adaptive_feedback(w_t, metrics):
    """
    metrics = {
        'H_avg': 平均熵,
        'conf_trend': 置信度趋势,
        'KL_div': 类别分布偏差,
        'drift': 特征漂移
    }
    """
    # 计算调整量
    Δw = 0

    # 1. 不确定性调整
    if metrics['H_avg'] > 0.8 * log(K):
        Δw += 0.1
    elif metrics['H_avg'] < 0.3 * log(K):
        Δw -= 0.05

    # 2. 趋势调整
    if metrics['conf_trend'] < -0.05:
        Δw += 0.15
    elif metrics['conf_trend'] > 0.05:
        Δw -= 0.05

    # 3. 漂移调整
    if metrics['drift'] > drift_threshold:
        Δw += 0.1

    # 4. 应用调整（带momentum）
    w_t_new = w_t + 0.7 * Δw  # momentum = 0.7
    w_t_new = np.clip(w_t_new, 0.0, 1.0)

    return w_t_new
```

**优点**：
- ✅ 综合多个信号，更稳健
- ✅ 使用momentum，避免震荡

**缺点**：
- ⚠️ 参数更多，需要调参

### 3.3 基于元学习的反馈（高级）

**核心思想**：学习一个反馈策略网络

```python
class FeedbackPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),  # 输入：6个监控指标
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # 输出：Δw ∈ [-1, 1]
        )

    def forward(self, metrics):
        return self.net(metrics)
```

**训练**：
- 在源域数据上模拟在线场景
- 使用强化学习（如PPO）训练策略网络
- 奖励函数：准确率提升

**何时使用**：
- MVP版本效果不佳
- 有足够的计算资源
- 作为ablation study的上界

---

## 4. 稳定性保证

### 4.1 平滑机制

**问题**：反馈可能导致对齐权重剧烈震荡

**解决方案1：滑动窗口平均**
```python
w_t_smooth = mean([w_i for i in range(t-5, t)])
```

**解决方案2：指数移动平均（EMA）**
```python
w_t_smooth = α * w_t + (1 - α) * w_{t-1}
```

**参数**：α = 0.3（较小的α更平滑）

**解决方案3：Momentum**
```python
v_t = β * v_{t-1} + (1 - β) * Δw_t
w_t = w_{t-1} + v_t
```

**参数**：β = 0.7

### 4.2 变化率限制

**问题**：单次调整幅度过大

**解决方案**：限制最大变化率
```python
Δw_max = 0.2  # 单次最多变化0.2
Δw_t = np.clip(Δw_t, -Δw_max, Δw_max)
```

### 4.3 死区（Dead Zone）

**问题**：微小的指标变化导致频繁调整

**解决方案**：设置死区
```python
if abs(H_avg - H_target) < ε:
    # 在死区内，不调整
    Δw = 0
else:
    # 在死区外，正常调整
    Δw = compute_adjustment(H_avg)
```

**参数**：ε = 0.1

---

## 5. 在线更新策略

### 5.1 何时更新？

**策略1：每个trial更新（实时）**
```python
for t, x_t in enumerate(trial_stream):
    # 计算指标
    metrics = compute_metrics(history[-W:])
    # 更新对齐权重
    w_t = adaptive_feedback(w_t, metrics)
```

**优点**：
- ✅ 响应快，适应性强

**缺点**：
- ⚠️ 可能不稳定，容易震荡

**策略2：每N个trial更新（批量）**
```python
for t, x_t in enumerate(trial_stream):
    if t % N == 0:
        # 每N个trial更新一次
        metrics = compute_metrics(history[-W:])
        w_t = adaptive_feedback(w_t, metrics)
```

**优点**：
- ✅ 更稳定，指标更可靠

**缺点**：
- ⚠️ 响应慢，可能错过快速变化

**建议**：
- MVP版本：N = 5（每5个trial更新）
- 完整版本：N = 1（每个trial更新，但使用平滑机制）

### 5.2 冷启动策略

**问题**：前几个trial没有足够的历史信息

**解决方案**：
```python
if t < W:
    # 冷启动阶段：使用默认权重
    w_t = 0.5
else:
    # 正常阶段：使用反馈
    w_t = adaptive_feedback(w_t, metrics)
```

### 5.3 重置机制

**问题**：如果性能持续下降，可能需要重置

**解决方案**：
```python
if conf_avg < conf_threshold_critical:
    # 性能严重下降，重置到初始状态
    w_t = 0.5
    μ_tgt = mean([x_i for i in range(t-W, t)])
    print(f"Reset at trial {t}")
```

**参数**：
- `conf_threshold_critical = 0.4`

---

## 6. 表征-行为桥接

### 6.1 显式建模表征-行为关系

**核心思想**：不仅对齐表征，还要确保行为一致

**方法1：联合优化**
```python
L_total = L_cls + λ_rep * L_rep + λ_beh * L_beh
```

其中：
- `L_cls`：分类损失
- `L_rep`：表征对齐损失（如CKA）
- `L_beh`：行为一致性损失

**行为一致性损失**：
```python
L_beh = ||acc_src - acc_tgt||^2
```

**直觉**：
- 不仅要表征相似，还要性能相似
- 显式优化表征-行为关系

**方法2：约束优化**
```python
# 约束：表征相似度 > threshold
# 目标：最大化准确率
maximize acc_tgt
subject to CKA(Φ_src, Φ_tgt) > 0.7
```

### 6.2 表征-行为一致性监控

**定义**：
```python
rep_beh_consistency = correlation(CKA_values, acc_values)
```

**使用**：
```python
if rep_beh_consistency < 0.5:
    # 表征-行为不一致 → 调整对齐策略
    # 可能需要减少对齐强度，保留更多个体特征
    w_t = max(w_t - 0.1, 0.0)
```

---

## 7. 实验验证

### 7.1 验证反馈有效性

**实验设计**：
```python
# Baseline: 无反馈（固定w_t）
# Ablation 1: 只有不确定性反馈
# Ablation 2: 只有趋势反馈
# 完整版本: 所有反馈规则
```

**评估指标**：
- 跨被试准确率
- Within-session稳定性（准确率方差）
- 对齐权重的变化轨迹

**预期结果**：
- 有反馈的方法在session后期性能更稳定
- 对齐权重应该随着session进行而调整

### 7.2 可视化分析

**图1：对齐权重演化**
```python
plt.plot(w_t_values)
plt.xlabel("Trial")
plt.ylabel("Alignment Weight")
plt.title("Evolution of Alignment Weight")
```

**预期**：
- w_t应该有明显的变化趋势
- 不应该是常数或随机震荡

**图2：性能 vs 对齐权重**
```python
plt.scatter(w_t_values, acc_values)
plt.xlabel("Alignment Weight")
plt.ylabel("Accuracy")
```

**预期**：
- 应该有一定的相关性
- 识别出最优的对齐权重范围

**图3：反馈触发时刻**
```python
# 标记反馈调整的时刻
plt.plot(acc_values)
plt.scatter(feedback_times, acc_values[feedback_times], color='red')
```

**预期**：
- 反馈应该在性能下降时触发
- 反馈后性能应该有改善

---

## 8. 调试技巧

### 8.1 检查反馈是否生效

**Step 1：记录反馈触发次数**
```python
feedback_count = {
    'uncertainty': 0,
    'trend': 0,
    'class_bias': 0
}

# 在反馈规则中
if H_avg > threshold:
    feedback_count['uncertainty'] += 1
```

**预期**：
- 每种反馈都应该触发若干次（不应该全是0）
- 如果某种反馈从不触发，说明阈值设置有问题

**Step 2：可视化监控指标**
```python
plt.subplot(3, 1, 1)
plt.plot(H_avg_values)
plt.title("Average Entropy")

plt.subplot(3, 1, 2)
plt.plot(conf_trend_values)
plt.title("Confidence Trend")

plt.subplot(3, 1, 3)
plt.plot(w_t_values)
plt.title("Alignment Weight")
```

**预期**：
- 监控指标应该有合理的变化
- w_t的变化应该与监控指标相关

### 8.2 检查稳定性

**Step 1：计算对齐权重的方差**
```python
w_variance = np.var(w_t_values)
```

**预期**：
- 方差不应该太大（< 0.1）
- 如果方差很大，说明震荡严重

**Step 2：检查相邻trial的变化**
```python
w_diff = np.diff(w_t_values)
plt.hist(w_diff)
```

**预期**：
- 大部分变化应该很小（< 0.1）
- 不应该有很多大的跳变（> 0.3）

---

## 9. 常见问题

### Q1: 反馈导致性能震荡怎么办？

**原因**：
- 反馈太激进
- 没有平滑机制

**解决方案**：
1. 减小调整步长（α, β）
2. 增加滑动窗口大小（W）
3. 使用EMA或momentum
4. 增加死区（ε）

### Q2: 反馈没有效果怎么办？

**原因**：
- 监控指标不敏感
- 阈值设置不合理
- 反馈规则太保守

**解决方案**：
1. 检查监控指标是否有变化
2. 调整阈值（可视化指标分布）
3. 增大调整步长
4. 尝试不同的反馈规则

### Q3: 如何选择合适的窗口大小W？

**原则**：
- W太小：指标不稳定，容易震荡
- W太大：响应慢，无法适应快速变化

**建议**：
- 从W=10开始
- 如果震荡严重，增大W
- 如果响应太慢，减小W

---

## 10. 下一步

完成行为引导反馈设计后，继续阅读：
- **04-implementation-guide.md**：开始编码实现
- **05-experiment-design.md**：设计完整的实验方案

---

**文档版本**: v1.0
**最后更新**: 2026-03-05
**状态**: 行为引导反馈设计完成
