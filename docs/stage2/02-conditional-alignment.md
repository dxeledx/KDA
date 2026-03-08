# 阶段二 - 条件对齐网络 (Conditional Alignment Network)

## 1. 核心设计理念

### 1.1 为什么需要条件对齐？

**传统对齐的局限性**：
- EA/RA使用固定的对齐参数，对所有trial一视同仁
- 无法适应within-session的动态变化（疲劳、注意力波动）
- 忽略了不同trial的个体差异

**条件对齐的优势**：
- **自适应性**：根据当前trial的特征动态调整对齐强度
- **上下文感知**：利用历史信息和几何特征
- **精细控制**：不同trial使用不同的对齐策略

### 1.2 设计原则

1. **简单优先**：MVP版本使用标量权重 w_t ∈ [0,1]
2. **可解释性**：上下文特征应该有明确的物理意义
3. **计算高效**：在线推理时间 < 10ms/trial
4. **稳定性**：避免对齐权重剧烈震荡

---

## 2. 对齐参数形式

### 2.1 MVP版本：标量权重

**数学形式**：
```
x'_t = (1 - w_t) · x_t + w_t · Align(x_t)
```

其中：
- `x_t ∈ ℝ^d`：当前trial的特征（CSP特征或协方差矩阵）
- `w_t ∈ [0, 1]`：对齐权重
- `Align(·)`：基础对齐方法（EA或RA）
- `x'_t`：对齐后的特征

**权重的物理意义**：
- `w_t = 0`：完全不对齐，保留原始特征（适用于接近源域的trial）
- `w_t = 1`：完全对齐，使用EA/RA特征（适用于远离源域的trial）
- `w_t = 0.5`：部分对齐，平衡原始和对齐特征

**优点**：
- ✅ 只有1个参数，易于优化
- ✅ 线性插值，计算高效
- ✅ 可解释性强
- ✅ 稳定性好

**缺点**：
- ⚠️ 表达能力有限，无法捕捉特征维度间的差异
- ⚠️ 假设所有维度使用相同的对齐强度

### 2.2 扩展版本1：对角矩阵权重

**数学形式**：
```
x'_t = (I - W_t) · x_t + W_t · Align(x_t)
```

其中：
- `W_t = diag(w_1, w_2, ..., w_d) ∈ ℝ^{d×d}`：对角权重矩阵
- 每个维度有独立的对齐权重

**优点**：
- ✅ 更灵活，可以对不同特征维度使用不同对齐强度
- ✅ 仍然保持计算高效（对角矩阵乘法）

**缺点**：
- ⚠️ 参数量增加到 d 个
- ⚠️ 需要更多数据训练

**何时使用**：
- MVP版本效果不佳时
- 发现不同CSP分量需要不同对齐强度时

### 2.3 扩展版本2：完整对齐矩阵

**数学形式**：
```
x'_t = M_t · x_t
```

其中：
- `M_t ∈ ℝ^{d×d}`：完整的对齐矩阵
- 可以学习任意线性变换

**优点**：
- ✅ 最大的表达能力
- ✅ 可以学习复杂的对齐模式

**缺点**：
- ⚠️ 参数量 = d²，容易过拟合
- ⚠️ 计算开销大
- ⚠️ 可解释性差

**何时使用**：
- 作为ablation study的上界
- 数据量充足时（>1000 trials）

---

## 3. 上下文设计 (Context Vector c_t)

### 3.1 设计原则

**上下文应该回答的问题**：
1. 当前trial离源域有多远？（几何距离）
2. 当前trial的预测有多不确定？（预测熵）
3. 最近的性能趋势如何？（置信度变化）

### 3.2 MVP版本：几何上下文

**特征1：到源域中心的距离**
```python
d_src = ||x_t - μ_src||_2
```

其中：
- `μ_src`：源域特征的均值（在离线阶段计算）
- 欧氏距离（如果在特征空间）或Riemannian距离（如果在SPD流形）

**直觉**：
- 距离大 → trial远离源域 → 需要更强对齐（w_t 大）
- 距离小 → trial接近源域 → 保留原始特征（w_t 小）

**特征2：到目标域中心的距离**
```python
d_tgt = ||x_t - μ_tgt||_2
```

其中：
- `μ_tgt`：目标域特征的均值（用最近N个trial估计）
- 使用滑动窗口更新：`μ_tgt = (1-α) · μ_tgt + α · x_t`

**直觉**：
- 距离大 → trial是outlier → 可能需要更强对齐
- 距离小 → trial符合目标域分布 → 可以减少对齐

**特征3：最近trial的方差**
```python
σ_recent = std([x_{t-N}, ..., x_{t-1}])
```

**直觉**：
- 方差大 → 信号不稳定（疲劳、注意力波动）→ 需要更强对齐
- 方差小 → 信号稳定 → 可以减少对齐

**MVP上下文向量**：
```python
c_t = [d_src, d_tgt, σ_recent]  # 3维向量
```

### 3.3 完整版本：几何 + 行为上下文

**特征4：预测熵**
```python
H_t = -Σ p_k log p_k
```

其中：
- `p_k`：分类器输出的类别概率
- 熵越高，预测越不确定

**直觉**：
- 高熵 → 模型不确定 → 需要更强对齐
- 低熵 → 模型确定 → 可以减少对齐

**特征5：平均置信度**
```python
conf_avg = mean([max(p_k^{(i)}) for i in range(t-N, t)])
```

**直觉**：
- 置信度下降 → 性能可能退化 → 需要调整对齐
- 置信度稳定 → 当前策略有效 → 保持对齐

**特征6：预测类别分布**
```python
class_dist = [count(ŷ=k) / N for k in classes]
KL_div = KL(class_dist || uniform_dist)
```

**直觉**：
- KL散度大 → 类别不平衡 → 可能有偏差 → 需要调整

**完整上下文向量**：
```python
c_t = [d_src, d_tgt, σ_recent, H_t, conf_avg, KL_div]  # 6维向量
```

### 3.4 上下文特征的归一化

**重要**：不同特征的量纲不同，需要归一化！

```python
# 方法1：Z-score归一化
c_t_norm = (c_t - μ_c) / σ_c

# 方法2：Min-Max归一化
c_t_norm = (c_t - c_min) / (c_max - c_min)
```

**建议**：
- 使用源域数据计算 μ_c, σ_c（离线阶段）
- 在线阶段使用相同的归一化参数

---

## 4. 条件网络结构

### 4.1 MVP版本：简单MLP

**网络结构**：
```python
class ConditionalAlignmentNetwork(nn.Module):
    def __init__(self, context_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出 w_t ∈ [0, 1]
        )

    def forward(self, c_t):
        return self.net(c_t)
```

**参数量**：
- 第1层：3 × 16 + 16 = 64
- 第2层：16 × 8 + 8 = 136
- 第3层：8 × 1 + 1 = 9
- **总计**：209个参数（非常小！）

**训练技巧**：
- Dropout防止过拟合
- 使用Adam优化器，学习率 1e-3
- Batch size = 32

### 4.2 更简单的版本：线性模型

如果数据量很少（<500 trials），可以用线性模型：

```python
w_t = sigmoid(a · d_src + b · d_tgt + c · σ_recent + d)
```

**参数量**：只有4个！

**优点**：
- ✅ 极简，不容易过拟合
- ✅ 可解释性强（可以看每个特征的权重）
- ✅ 训练快

**缺点**：
- ⚠️ 表达能力有限

### 4.3 扩展版本：注意力机制

如果需要更强的表达能力：

```python
class AttentionConditionalNetwork(nn.Module):
    def __init__(self, context_dim=6):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=context_dim, num_heads=2)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, c_t, history_contexts):
        # 使用历史上下文作为key和value
        attn_output, _ = self.attention(c_t, history_contexts, history_contexts)
        return self.mlp(attn_output)
```

**何时使用**：
- MVP版本效果不佳
- 需要利用长期历史信息

---

## 5. 训练策略

### 5.1 离线训练

**目标**：在源域数据上训练条件网络 g(·)

**训练数据构造**：
```python
# 对每个源域trial
for x_i in source_data:
    # 1. 计算上下文
    c_i = compute_context(x_i, source_data)

    # 2. 预测对齐权重
    w_i = g(c_i)

    # 3. 对齐特征
    x_i_aligned = (1 - w_i) * x_i + w_i * Align(x_i)

    # 4. 分类
    y_pred = classifier(x_i_aligned)

    # 5. 计算损失
    loss = CrossEntropy(y_pred, y_true)
```

**损失函数**：
```python
L_total = L_cls + λ_smooth * L_smooth + λ_reg * L_reg
```

其中：
- `L_cls`：分类损失（交叉熵）
- `L_smooth`：平滑损失（防止w_t剧烈变化）
- `L_reg`：正则化损失（防止过拟合）

**平滑损失**：
```python
L_smooth = Σ ||w_i - w_{i-1}||^2
```

**直觉**：相邻trial的对齐权重不应该差异太大

**正则化损失**：
```python
L_reg = ||θ||^2  # L2正则化
```

**超参数**：
- `λ_smooth = 0.1`（平滑权重）
- `λ_reg = 1e-4`（正则化权重）

### 5.2 联合训练 vs 两阶段训练

**方法1：两阶段训练（推荐）**
```python
# 阶段1：训练基础分类器
classifier = train_CSP_LDA(source_data)

# 阶段2：固定分类器，训练条件网络
g = train_conditional_network(source_data, classifier)
```

**优点**：
- ✅ 简单，易于调试
- ✅ 可以复用阶段一的分类器

**方法2：联合训练**
```python
# 同时优化分类器和条件网络
classifier, g = train_jointly(source_data)
```

**优点**：
- ✅ 理论上可以达到更好的性能

**缺点**：
- ⚠️ 训练不稳定
- ⚠️ 需要更多调参

**建议**：MVP版本使用两阶段训练

### 5.3 数据增强

**问题**：源域数据可能不足以训练条件网络

**解决方案**：模拟目标域分布

```python
# 方法1：添加噪声
x_aug = x + ε, ε ~ N(0, σ^2)

# 方法2：Mixup
x_aug = α * x_i + (1 - α) * x_j, α ~ Beta(0.5, 0.5)

# 方法3：时间抖动
x_aug = shift(x, Δt), Δt ~ Uniform(-50ms, 50ms)
```

**建议**：
- 先不用数据增强，看MVP效果
- 如果过拟合严重，再加数据增强

---

## 6. 在线推理

### 6.1 推理流程

```python
def predict_online(trial_stream, g, classifier):
    # 初始化
    history = []
    μ_tgt = None  # 目标域中心（初始为None）

    for t, x_t in enumerate(trial_stream):
        # 1. 计算上下文
        if t < 10:
            # 前10个trial：只用几何距离
            c_t = [d_src(x_t), 0, 0]
        else:
            # 后续trial：使用完整上下文
            c_t = compute_context(x_t, history)

        # 2. 预测对齐权重
        w_t = g(c_t)

        # 3. 对齐特征
        x_t_aligned = (1 - w_t) * x_t + w_t * Align(x_t)

        # 4. 分类
        y_pred, conf = classifier.predict_proba(x_t_aligned)

        # 5. 更新历史
        history.append({
            'x': x_t,
            'y_pred': y_pred,
            'conf': conf,
            'w': w_t
        })

        # 6. 更新目标域中心（滑动窗口）
        if μ_tgt is None:
            μ_tgt = x_t
        else:
            μ_tgt = 0.9 * μ_tgt + 0.1 * x_t

        yield y_pred
```

### 6.2 冷启动问题

**问题**：前几个trial没有历史信息，上下文不完整

**解决方案**：
1. **使用默认对齐权重**：`w_t = 0.5`（前10个trial）
2. **只使用几何距离**：`c_t = [d_src, 0, 0]`
3. **逐步引入行为特征**：从trial 10开始使用完整上下文

### 6.3 计算效率

**时间复杂度**：
- 上下文计算：O(d)
- 条件网络前向传播：O(d × h)（h是隐藏层大小）
- 对齐：O(d²)（如果用RA）或O(d)（如果用EA）
- 分类：O(d)

**总计**：O(d²)（主要瓶颈在对齐）

**优化**：
- 预计算源域中心 μ_src
- 使用EA而不是RA（更快）
- 批量处理（如果允许延迟）

---

## 7. 调试和验证

### 7.1 验证清单

**Step 1：检查上下文特征**
```python
# 可视化上下文特征的分布
plt.hist(d_src_values)
plt.title("Distribution of d_src")
```

**预期**：
- d_src应该有合理的范围（不要全是0或全是inf）
- 不同被试的d_src分布应该有差异

**Step 2：检查对齐权重**
```python
# 可视化对齐权重的变化
plt.plot(w_t_values)
plt.title("Alignment weight over time")
```

**预期**：
- w_t应该在[0, 1]范围内
- w_t不应该全是0或全是1（说明网络没学到东西）
- w_t应该有一定的变化（不应该是常数）

**Step 3：检查对齐效果**
```python
# 对比对齐前后的特征分布
plt.scatter(x_before[:, 0], x_before[:, 1], label="Before")
plt.scatter(x_after[:, 0], x_after[:, 1], label="After")
```

**预期**：
- 对齐后的特征应该更接近源域
- 但不应该完全重合（保留个体差异）

### 7.2 消融研究

**实验1：固定权重 vs 动态权重**
```python
# Baseline 1: w_t = 0 (不对齐)
# Baseline 2: w_t = 0.5 (固定部分对齐)
# Baseline 3: w_t = 1 (完全对齐)
# 你的方法: w_t = g(c_t) (动态对齐)
```

**预期**：动态对齐应该比所有固定权重都好

**实验2：不同上下文特征的贡献**
```python
# Ablation 1: 只用 d_src
# Ablation 2: 只用 d_tgt
# Ablation 3: 只用 σ_recent
# 完整版本: 使用所有特征
```

**预期**：识别出最重要的上下文特征

---

## 8. 常见问题

### Q1: 对齐权重总是接近0或1怎么办？

**原因**：
- 网络学到了"全对齐"或"全不对齐"是最优策略
- 可能是上下文特征不够区分

**解决方案**：
1. 检查上下文特征是否有区分度
2. 添加正则化：`L_reg = (w_t - 0.5)^2`（鼓励中间值）
3. 尝试不同的网络结构

### Q2: 对齐权重震荡怎么办？

**原因**：
- 没有平滑损失
- 学习率太大

**解决方案**：
1. 增大 λ_smooth
2. 降低学习率
3. 使用滑动平均：`w_t = 0.8 * w_{t-1} + 0.2 * g(c_t)`

### Q3: 条件网络过拟合怎么办？

**原因**：
- 网络太复杂
- 数据太少

**解决方案**：
1. 使用更简单的网络（甚至线性模型）
2. 增大dropout
3. 使用数据增强
4. 使用leave-one-subject-out验证

---

## 9. 下一步

完成条件对齐网络设计后，继续阅读：
- **03-behavior-guided-feedback.md**：设计行为引导反馈机制
- **04-implementation-guide.md**：开始编码实现

---

**文档版本**: v1.0
**最后更新**: 2026-03-05
**状态**: 条件对齐网络设计完成
