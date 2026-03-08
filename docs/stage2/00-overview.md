# 阶段二：DCA-BGF方法设计与实现 - 总览

## 📋 目标

1. **设计DCA-BGF方法**：Dynamic Conditional Alignment with Behavior-Guided Feedback
2. **实现MVP版本**：最小可行产品，验证核心思路
3. **初步实验验证**：在阶段一的baseline基础上验证提升

---

## 📂 文档结构

### [00 - 总览](00-overview.md)
- 阶段目标
- 方法概述
- 预计时间
- 完成标准

### [01 - 方法设计](01-method-design.md)
- 核心创新点
- 算法框架
- 数学形式化
- 与baseline的对比

### [02 - 条件对齐网络](02-conditional-alignment.md)
- 上下文设计 (c_t)
- 对齐参数形式 (θ_t)
- 网络结构
- 训练策略

### [03 - 行为引导反馈](03-behavior-guided-feedback.md)
- 监控指标设计
- 反馈规则
- 在线更新策略
- 稳定性保证

### [04 - 实现指南](04-implementation-guide.md)
- 代码结构
- 模块接口
- 实现步骤
- 调试技巧

### [05 - 实验设计](05-experiment-design.md)
- 实验协议
- 消融研究
- 对比方法
- 评估指标

### [06 - 预期结果](06-expected-results.md)
- 性能目标
- 关键图表
- 失败预案
- 风险对冲

---

## 🎯 核心思路

### 问题
阶段一发现：
- ✅ 表征-行为不一致现象存在 (r=0.39)
- ✅ 静态对齐有效但有限 (EA/RA: r=0.64)
- ⚠️ 跨被试性能仍然较低 (38-43%)

### 解决方案：DCA-BGF

**核心创新**：
1. **条件对齐**：根据当前trial的上下文动态预测对齐参数
2. **行为引导**：利用在线性能指标调整对齐策略
3. **闭环适应**：持续自我修正，适应within-session动态

**简化版本**（MVP）：
```python
# 离线训练
g = ConditionalAlignmentNetwork()  # 预测对齐权重 w_t ∈ [0,1]
g.fit(source_data)

# 在线推理
for trial in target_data:
    # 1. 计算上下文
    c_t = compute_context(trial, recent_history)

    # 2. 预测对齐权重
    w_t = g.predict(c_t)

    # 3. 部分对齐
    trial_aligned = (1-w_t) * trial + w_t * align(trial)

    # 4. 分类
    y_pred = classifier(trial_aligned)

    # 5. 行为引导反馈
    if uncertainty_high(y_pred):
        w_t *= 1.2  # 增强对齐
```

---

## ⏱️ 预计时间

- **Week 1-2**: 方法设计和数学推导
- **Week 3-4**: MVP实现和调试
- **Week 5-6**: 完整实现和实验
- **Week 7**: 消融研究和分析

**总计**: 6-7周

---

## ✅ 完成标准

### 最小成功标准
- ✅ MVP版本实现并运行
- ✅ 在至少3个被试上比RA好
- ✅ 表征-行为相关性 r > 0.70
- ✅ 代码可复现

### 目标成功标准
- ✅ 跨被试准确率比RA提升 +3-5%
- ✅ 表征-行为相关性 r > 0.75
- ✅ 在session后期性能更稳定
- ✅ 完整的消融研究

### 理想成功标准
- ✅ 跨被试准确率比RA提升 +5-10%
- ✅ 表征-行为相关性 r > 0.80
- ✅ 在所有被试上都有提升
- ✅ 计算效率可接受（<2x RA）

---

## 🚀 快速开始

### 1. 阅读顺序（推荐）
```
00-overview.md (本文件)
    ↓
01-method-design.md (理解核心思路)
    ↓
02-conditional-alignment.md (设计条件对齐)
    ↓
03-behavior-guided-feedback.md (设计行为反馈)
    ↓
04-implementation-guide.md (开始编码)
    ↓
05-experiment-design.md (设计实验)
```

### 2. 最小可行路径（MVP优先）
```
只实现：
- 标量对齐权重 w_t ∈ [0,1]
- 简单上下文：几何距离
- 基础反馈：不确定性调整
- 在3个被试上验证

跳过：
- 复杂的对齐参数（仿射变换、MLP）
- 复杂的上下文（熵、KL散度）
- 复杂的反馈规则
```

### 3. 完整路径
```
按顺序完成所有文档
实现完整的DCA-BGF
全面的实验验证
```

---

## 📊 与阶段一的衔接

### 复用的代码
```python
from src.data.loader import BCIDataLoader  # 数据加载
from src.features.csp import CSP  # 特征提取
from src.alignment.riemannian import RiemannianAlignment  # 基础对齐
from src.evaluation.metrics import cka, compute_metrics  # 评估
```

### 新增的模块
```python
src/alignment/conditional.py  # 条件对齐网络
src/alignment/behavior_feedback.py  # 行为引导反馈
src/alignment/dca_bgf.py  # 完整的DCA-BGF
experiments/dca_bgf_mvp.py  # MVP实验
experiments/dca_bgf_full.py  # 完整实验
```

---

## 🎓 理论基础

### 需要的数学知识
- ✅ 线性代数（矩阵运算）
- ✅ 概率论（熵、KL散度）
- ✅ 优化理论（梯度下降）
- 🔶 在线学习（可选，用于理论分析）

### 需要的编程技能
- ✅ Python + NumPy
- ✅ PyTorch（如果用神经网络）
- 🔶 scikit-learn（已经在用）

---

## 🤔 常见问题

### Q1: 我必须用神经网络吗？
**A**: 不是！MVP版本可以用简单的线性模型：
```python
w_t = sigmoid(a * d_geo + b)  # 只有2个参数
```

### Q2: 如果MVP效果不好怎么办？
**A**:
1. 先检查是否正确实现
2. 调整上下文特征
3. 尝试不同的对齐权重范围
4. 参考文档中的"失败预案"

### Q3: 我需要GPU吗？
**A**:
- MVP版本：不需要
- 完整版本（如果用MLP）：建议有GPU，但CPU也可以

### Q4: 阶段二需要多久？
**A**:
- 最快：2-3周（只做MVP）
- 推荐：4-6周（MVP + 完整版本）
- 完整：6-8周（包括所有消融研究）

---

## 📞 需要帮助？

### 设计阶段
- 不确定上下文特征怎么设计？→ 看 `02-conditional-alignment.md`
- 不确定反馈规则怎么写？→ 看 `03-behavior-guided-feedback.md`

### 实现阶段
- 代码不知道怎么组织？→ 看 `04-implementation-guide.md`
- 遇到bug？→ 参考调试技巧

### 实验阶段
- 不知道做什么实验？→ 看 `05-experiment-design.md`
- 结果不理想？→ 看 `06-expected-results.md` 的失败预案

---

## 🎯 成功的关键

1. **从简单开始**：MVP优先，不要一开始就做复杂版本
2. **快速迭代**：实现→测试→改进，不要等到完美才测试
3. **对比baseline**：每次改进都和RA对比，确保有提升
4. **记录实验**：记录所有尝试，包括失败的
5. **保持信心**：即使提升不大（+2-3%），也是有价值的创新

---

**准备好了吗？让我们开始设计DCA-BGF方法！** 🚀

**下一步**：阅读 `01-method-design.md` 了解详细的方法设计。
