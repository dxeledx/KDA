# 阶段二：DCA-BGF方法设计与实现 - 完整指南

欢迎来到 DCA-BGF 项目的阶段二！本阶段的目标是设计和实现 **Dynamic Conditional Alignment with Behavior-Guided Feedback (DCA-BGF)** 方法，在阶段一的baseline基础上实现创新突破。

---

## 📚 文档导航

### [00 - 总览](00-overview.md) ⭐ 必读
- 阶段目标和完成标准
- 方法概述
- 时间安排
- 快速开始指南

### [01 - 方法设计](01-method-design.md) ⭐ 必读
- 核心创新点（条件对齐 + 行为反馈）
- 算法框架和数学形式化
- 与baseline的对比
- MVP版本简化策略

### [02 - 条件对齐网络](02-conditional-alignment.md)
- 对齐参数形式（标量权重 vs 矩阵）
- 上下文设计（几何特征 + 行为特征）
- 网络结构（MLP vs 线性模型）
- 训练策略（两阶段 vs 联合训练）

### [03 - 行为引导反馈](03-behavior-guided-feedback.md)
- 监控指标设计（熵、置信度、类别分布）
- 反馈规则（不确定性、趋势、偏差）
- 稳定性保证（平滑、限制、死区）
- 表征-行为桥接

### [04 - 实现指南](04-implementation-guide.md) ⭐ 必读
- 代码结构和模块组织
- 核心模块实现（含完整代码）
- 实验脚本模板
- 调试技巧和单元测试

### [05 - 实验设计](05-experiment-design.md)
- 实验协议（LOSO, Within-Subject）
- 对比方法和消融研究
- 评估指标和统计检验
- 完整实验流程

### [06 - 预期结果与风险对冲](06-expected-results.md) ⭐ 必读
- 性能目标（准确率、表征-行为一致性）
- 风险分析和应对策略
- 成功标准（最小/目标/理想）
- 投稿策略和时间线

### [07 - 失败分析与阶段三 Pivot](07-failure-analysis-and-stage3-pivot.md) ⭐ 最新结论
- rescue 实验的最终判断
- 哪些假设已被证伪 / 保留
- 为什么阶段三应转向 RA-first safe adaptation
- 下一阶段的具体研究建议

---

## 🎯 核心思路

### 问题
阶段一发现：
- ✅ 表征-行为不一致现象存在 (r=0.39)
- ✅ 静态对齐有效但有限 (EA/RA: r=0.64)
- ⚠️ 跨被试性能仍然较低 (38-43%)

### 解决方案：DCA-BGF

**三大创新**：
1. **条件对齐**：根据当前trial的上下文动态预测对齐参数
2. **行为引导**：利用在线性能指标调整对齐策略
3. **闭环适应**：持续自我修正，适应within-session动态

**MVP版本**（最简单，推荐先实现）：
```python
# 离线训练
g = ConditionalAlignmentNetwork(context_dim=3)  # 只用几何特征
g.fit(source_data)

# 在线推理
for trial in target_data:
    # 1. 计算上下文（几何距离）
    c_t = [d_src, d_tgt, σ_recent]

    # 2. 预测对齐权重
    w_t = g.predict(c_t)  # w_t ∈ [0, 1]

    # 3. 部分对齐
    trial_aligned = (1-w_t) * trial + w_t * EA(trial)

    # 4. 分类
    y_pred = classifier(trial_aligned)

    # 5. 行为反馈（简单规则）
    if entropy(y_pred) > threshold:
        w_t += 0.1  # 增强对齐
```

---

## ⏱️ 时间安排

| 阶段 | 时间 | 任务 | 检查点 |
|------|------|------|--------|
| **Week 1-2** | MVP实现 | 实现条件对齐网络和行为反馈 | MVP在1个被试上比RA好 |
| **Week 3-4** | 完整实验 | LOSO实验（9个被试）+ 消融研究 | 平均准确率 > 63% |
| **Week 5** | 分析 | 表征-行为分析 + 统计检验 | r > 0.70 |
| **Week 6** | 完善 | 补充实验 + 代码整理 | 所有实验完成 |

**总计**：6-7周

---

## ✅ 完成标准

### 最小成功标准（必须达到）
- ✅ MVP版本实现并运行
- ✅ 跨被试准确率 > 63%（比RA提升 > 1%）
- ✅ 表征-行为相关性 r > 0.70
- ✅ 统计显著性 p < 0.05

### 目标成功标准（期望达到）
- ✅ 跨被试准确率 > 68%（比RA提升 > 5%）
- ✅ 表征-行为相关性 r > 0.75
- ✅ Within-session稳定性提升 > 20%
- ✅ 完整的消融研究

### 理想成功标准（最好达到）
- ✅ 跨被试准确率 > 70%（比RA提升 > 8%）
- ✅ 表征-行为相关性 r > 0.80
- ✅ 在所有9个被试上都有提升
- ✅ 在线推理时间 < 10ms/trial

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
    ↓
06-expected-results.md (了解预期和风险)
```

### 2. 最小可行路径（MVP优先）
```
只实现：
- 标量对齐权重 w_t ∈ [0,1]
- 简单上下文：几何距离 [d_src, d_tgt, σ_recent]
- 基础反馈：不确定性调整
- 在3个被试上验证

跳过：
- 复杂的对齐参数（仿射变换、MLP）
- 复杂的上下文（熵、KL散度）
- 复杂的反馈规则
```

### 3. 实现步骤
```
Step 1: 搭建代码结构（参考04-implementation-guide.md）
Step 2: 实现上下文计算（context.py）
Step 3: 实现条件对齐网络（conditional.py）
Step 4: 实现行为反馈（behavior_feedback.py）
Step 5: 整合为完整系统（dca_bgf.py）
Step 6: 编写实验脚本（dca_bgf_mvp.py）
Step 7: 在1个被试上测试
Step 8: 在所有被试上运行LOSO
```

---

## 📊 预期结果

### 性能目标

| 方法 | 准确率 | 表征-行为相关性 |
|------|--------|----------------|
| CSP+LDA | 40-50% | - |
| EA | 55-65% | r ≈ 0.64 |
| RA | 60-70% | r ≈ 0.64 |
| **DCA-BGF** | **65-75%** | **r > 0.75** |

### 关键发现

1. **条件对齐优于固定对齐**：动态权重比所有固定权重都好
2. **行为反馈提升稳定性**：session后期性能更稳定
3. **几何距离最重要**：d_src贡献最大
4. **表征-行为一致性提升**：r从0.64提升到0.75+

---

## 🚨 风险与应对

### 风险1：性能提升不显著
**应对**：
- 简化网络（防止过拟合）
- 改进上下文特征
- 换基础对齐方法（EA → RA）
- 降级目标（强调其他贡献）

### 风险2：表征-行为一致性没有提升
**应对**：
- 改进相似度计算（CKA → CCA）
- 改进行为一致性定义
- 增加样本量（其他数据集）
- 弱化这个贡献点

### 风险3：行为反馈不稳定
**应对**：
- 增加平滑机制（EMA, momentum）
- 降低反馈强度（减小步长）
- 改进监控指标（更稳健）
- 简化反馈规则

### 风险4：计算效率太低
**应对**：
- 优化网络（更小/线性模型）
- 优化对齐（EA而不是RA）
- 优化上下文计算（缓存、增量更新）
- 代码优化（向量化、GPU）

详细应对策略见 `06-expected-results.md`

---

## 📞 需要帮助？

### 设计阶段
- 不确定上下文特征怎么设计？→ 看 `02-conditional-alignment.md`
- 不确定反馈规则怎么写？→ 看 `03-behavior-guided-feedback.md`

### 实现阶段
- 代码不知道怎么组织？→ 看 `04-implementation-guide.md`
- 遇到bug？→ 参考调试技巧和单元测试

### 实验阶段
- 不知道做什么实验？→ 看 `05-experiment-design.md`
- 结果不理想？→ 看 `06-expected-results.md` 的风险应对

---

## 🎓 关键资源

### 理论基础
- 在线学习：Zinkevich 2003, "Online Convex Programming"
- 域适应：Ben-David et al. 2010, "A theory of learning from different domains"
- 表征相似度：Kornblith et al. 2019, "Similarity of Neural Network Representations Revisited"

### 代码参考
- PyRiemann：https://pyriemann.readthedocs.io/
- MNE-Python：https://mne.tools/
- PyTorch：https://pytorch.org/

### 数据集
- BCI Competition IV 2a：http://www.bbci.de/competition/iv/desc_2a.pdf

---

## 🎯 成功的关键

1. **从简单开始**：MVP优先，不要一开始就做复杂版本
2. **快速迭代**：实现→测试→改进，不要等到完美才测试
3. **对比baseline**：每次改进都和RA对比，确保有提升
4. **记录实验**：记录所有尝试，包括失败的
5. **保持信心**：即使提升不大（+2-3%），也是有价值的创新

---

## 📈 与阶段一的衔接

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
src/utils/context.py  # 上下文计算
src/utils/monitoring.py  # 监控指标
experiments/dca_bgf_mvp.py  # MVP实验
experiments/dca_bgf_full.py  # 完整实验
```

---

## 🚀 下一步

完成阶段二后，你将进入**阶段三：研究重构与论文主线重写**，先明确新的问题定义，再决定是否进入正式论文撰写。

阶段三的重点：
- 基于 `07-failure-analysis-and-stage3-pivot.md` 重写问题定义
- 从“连续动态 gating”转向“RA-first safe adaptation”
- 重新设定 acceptance criteria 和主实验
- 在新主线成立后再进入正式论文撰写

---

**准备好了吗？让我们开始设计和实现DCA-BGF方法！** 🎯

**建议**：
1. 先花1-2天仔细阅读所有文档
2. 理解核心思路后再开始编码
3. 从MVP版本开始，不要一开始就做复杂版本
4. 遇到问题随时参考文档中的调试技巧和风险应对策略

**祝你顺利完成阶段二！有任何问题随时呼叫我。** 🚀

---

**文档版本**: v1.0
**最后更新**: 2026-03-05
**状态**: 阶段二文档全部完成，可以开始实现
