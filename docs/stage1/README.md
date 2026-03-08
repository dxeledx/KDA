# 阶段一：基线方法复现 - 完整指南

欢迎来到 DCA-BGF 项目的阶段一！本阶段的目标是复现基线方法，验证"表征-行为不一致"现象，为后续的创新方法打下基础。

---

## 📚 文档导航

### [00 - 总览](00-overview.md)
- 阶段目标
- 文档结构
- 预计时间
- 完成标准

### [01 - 数据准备与预处理](01-data-preparation.md)
- BCI Competition IV Dataset 2a 下载
- 预处理流程（滤波、Epoch提取）
- 数据验证方法
- **预计时间**: 2-3天

### [02 - 基线方法：传统方法](02-baselines-traditional.md)
- Baseline 1: CSP + LDA
- Baseline 2: Euclidean Alignment + CSP + LDA
- Baseline 3: Riemannian Alignment + CSP + LDA
- **预计时间**: 5-7天

### [03 - 基线方法：深度学习](03-baselines-deep-learning.md) ⚠️ 未创建
- Baseline 4: CORAL
- Baseline 5: OTTA
- **预计时间**: 3-5天（可选）

### [04 - 现象验证](04-phenomenon-verification.md)
- 表征-行为不一致现象验证
- CKA相似度计算
- 可视化方法
- **预计时间**: 2-3天

### [05 - 评估与分析](05-evaluation-metrics.md) ⚠️ 未创建
- 评估指标详解
- 统计显著性检验
- 报告格式
- **预计时间**: 1-2天

### [06 - 调试指南](06-debugging-guide.md)
- 数据验证清单
- Baseline验证清单
- 常见问题和解决方案
- **随时参考**

### [07 - 代码组织](07-code-organization.md)
- 目录结构建议
- 依赖包清单
- 配置文件示例
- **开始前必读**

---

## 🎯 快速开始

### 1. 阅读顺序（推荐）
```
00-overview.md
    ↓
07-code-organization.md  (搭建项目结构)
    ↓
01-data-preparation.md   (准备数据)
    ↓
02-baselines-traditional.md  (复现baseline)
    ↓
04-phenomenon-verification.md  (验证现象)
    ↓
06-debugging-guide.md  (遇到问题时参考)
```

### 2. 最小可行路径（时间紧张）
```
只实现：
- 数据准备 (01)
- CSP+LDA (02的第一部分)
- RA+CSP+LDA (02的第三部分)
- 现象验证 (04)

跳过：
- EA (02的第二部分)
- CORAL/OTTA (03)
- 详细评估 (05)
```

### 3. 完整路径（推荐）
```
按顺序完成所有文档
预计总时间：2-4周
```

---

## ✅ 检查清单

### Week 1: 数据和基础Baseline
- [ ] 数据已下载 (01)
- [ ] 预处理pipeline已实现 (01)
- [ ] CSP+LDA已实现并验证 (02)
- [ ] Within-subject准确率 > 70% (02)

### Week 2: 对齐方法
- [ ] EA已实现 (02)
- [ ] RA已实现 (02)
- [ ] Cross-subject准确率：EA 55-65%, RA 60-70% (02)

### Week 3: 现象验证
- [ ] 表征相似度（CKA）已计算 (04)
- [ ] 行为一致性已计算 (04)
- [ ] 表征-行为散点图已生成 (04)
- [ ] 相关系数 r = 0.3-0.5 (04)

### Week 4: 完善和调试
- [ ] 所有代码已模块化 (07)
- [ ] 单元测试通过 (07)
- [ ] 结果已保存和可视化 (04)
- [ ] 准备进入阶段二

---

## 📊 预期结果

完成阶段一后，你应该有：

### 1. 数据
```
data/
├── raw/              # 原始.gdf文件
└── processed/        # 预处理后的.npy文件
```

### 2. 代码
```
src/
├── data/             # 数据加载和预处理
├── features/         # CSP特征提取
├── alignment/        # EA/RA对齐
├── models/           # LDA分类器
└── evaluation/       # 评估和可视化
```

### 3. 结果
```
results/
├── baselines/
│   ├── csp_lda/      # Baseline 1结果
│   ├── ea/           # Baseline 2结果
│   └── ra/           # Baseline 3结果
└── figures/
    ├── rep_beh_scatter.pdf          # 核心图：表征-行为散点图
    ├── covariance_heatmaps.pdf      # 协方差矩阵可视化
    └── transfer_matrix.pdf          # 跨被试性能热图
```

### 4. 性能基准
| 方法 | Within-Subject | Cross-Subject (LOSO) |
|------|----------------|---------------------|
| CSP+LDA | 70-80% | 40-50% |
| EA+CSP+LDA | - | 55-65% |
| RA+CSP+LDA | - | 60-70% |

### 5. 现象验证
- ✅ 表征-行为相关系数：r = 0.3-0.5
- ✅ 识别出2-3个典型不一致案例
- ✅ 证明了研究动机的合理性

---

## 🚨 常见问题

### Q1: 我应该先看哪个文档？
**A**: 先看 `00-overview.md` 了解全局，然后看 `07-code-organization.md` 搭建项目结构。

### Q2: 我可以跳过某些baseline吗？
**A**: 可以。最小路径是：CSP+LDA + RA + 现象验证。EA和CORAL/OTTA可以跳过。

### Q3: 我遇到bug怎么办？
**A**: 先查看 `06-debugging-guide.md`，运行验证脚本。如果还是解决不了，记录错误信息后呼叫我review。

### Q4: 我的结果和预期不符怎么办？
**A**:
1. 检查数据加载是否正确（形状、标签）
2. 检查预处理参数（滤波频段、时间窗口）
3. 先在within-subject上验证（应该>70%）
4. 参考 `06-debugging-guide.md` 的调试清单

### Q5: 我需要GPU吗？
**A**:
- CSP+LDA, EA, RA：不需要GPU
- CORAL, OTTA：需要GPU（但这两个是可选的）

---

## 📞 需要帮助？

### 代码完成后
1. 运行 `06-debugging-guide.md` 中的验证脚本
2. 确保所有检查通过
3. 呼叫我进行code review

### Review时提供
- 代码结构截图
- 关键结果（准确率、图表）
- 遇到的主要问题和解决方案

---

## 🎓 学习资源

### 推荐阅读
- BCI Competition IV 官方文档：http://www.bbci.de/competition/iv/desc_2a.pdf
- CSP原始论文：Ramoser et al. (2000)
- Riemannian Alignment：Zanini et al. (2018)
- CKA论文：Kornblith et al. (2019)

### 代码参考
- MNE-Python：https://mne.tools/
- PyRiemann：https://pyriemann.readthedocs.io/
- MOABB：https://moabb.neurotechx.com/

---

## 🚀 下一步

完成阶段一后，你将进入**阶段二：方法设计与实现**，开始开发你的 DCA-BGF (Dynamic Conditional Alignment with Behavior-Guided Feedback) 方法。

阶段二的重点：
- 设计条件对齐网络
- 实现行为引导反馈机制
- 在baseline基础上验证创新

---

**祝你顺利完成阶段一！有任何问题随时呼叫我。** 🎯
