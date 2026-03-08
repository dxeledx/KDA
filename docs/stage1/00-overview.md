# 阶段一：基线方法复现 - 总览

## 📋 目标

1. **验证"表征-行为不一致"现象**：用数据证明你的研究动机
2. **建立性能基准**：复现3-5个代表性方法，为你的创新提供对比
3. **熟悉数据和流程**：为后续方法开发打好基础

---

## 📂 文档结构

本阶段文档分为以下几个部分：

### 1. 数据准备 (`01-data-preparation.md`)
- 数据集下载和说明
- 预处理流程
- 数据验证方法

### 2. 基线方法 - 传统方法 (`02-baselines-traditional.md`)
- Baseline 1: CSP + LDA
- Baseline 2: Euclidean Alignment + CSP + LDA
- Baseline 3: Riemannian Alignment + CSP + LDA

### 3. 基线方法 - 深度学习 (`03-baselines-deep-learning.md`)
- Baseline 4: CORAL
- Baseline 5: OTTA (在线测试时适应)

### 4. 现象验证 (`04-phenomenon-verification.md`)
- 表征-行为不一致现象的验证实验
- CKA相似度计算
- 可视化方法

### 5. 评估与分析 (`05-evaluation-metrics.md`)
- 评估指标详解
- 统计显著性检验
- 报告格式

### 6. 调试指南 (`06-debugging-guide.md`)
- 数据验证清单
- Baseline验证清单
- 常见问题和解决方案

### 7. 代码组织 (`07-code-organization.md`)
- 目录结构建议
- 依赖包清单
- 配置文件示例

---

## ⏱️ 预计时间

- **Week 1**: 数据准备 + CSP+LDA baseline
- **Week 2**: EA/RA baseline + 现象验证
- **Week 3-4**: CORAL/OTTA (可选) + 完整评估

---

## ✅ 完成标准

阶段一完成后，你应该有：

1. **数据**：
   - ✅ BCI Competition IV Dataset 2a 已下载
   - ✅ 预处理pipeline已实现并验证
   - ✅ 数据质量检查通过

2. **Baseline结果**：
   - ✅ CSP+LDA: Within-subject 70-80%, Cross-subject 40-50%
   - ✅ EA: Cross-subject 55-65%
   - ✅ RA: Cross-subject 60-70%
   - ✅ (可选) CORAL/OTTA: Cross-subject 65-75%

3. **现象验证**：
   - ✅ 表征-行为散点图已生成
   - ✅ 相关系数 r = 0.3-0.5 (无对齐)
   - ✅ 识别出典型的不一致案例

4. **代码**：
   - ✅ 代码结构清晰，模块化
   - ✅ 所有baseline可复现
   - ✅ 结果已保存并可视化

---

## 🚀 下一步

完成阶段一后，进入**阶段二：方法设计与实现**，开始开发你的 DCA-BGF 方法。

---

## 📞 需要帮助？

如果在复现过程中遇到问题：
1. 先查看 `06-debugging-guide.md`
2. 检查数据和代码是否符合文档要求
3. 完成代码后，呼叫我进行 review
