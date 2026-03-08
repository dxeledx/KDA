# KSDA: Koopman-Space Dynamic Alignment

## 核心思想

将跨被试对齐从**协方差流形空间**转移到**Koopman线性空间**，利用线性结构实现更高效的动态适应，并用在线性能反馈修正Koopman算子估计。

---

## 三个递进的创新

### 1. Koopman空间对齐
**问题**: 协方差空间对齐需要复杂的SPD流形运算
**方案**: 在Koopman线性空间做对齐，计算更高效，动力学一致性天然嵌入

### 2. 线性结构的动态适应
**问题**: 协方差空间的在线更新需要Riemannian几何
**方案**: Koopman空间是线性的，可用标准在线学习算法，理论分析更清晰

### 3. 性能驱动的Koopman修正
**问题**: 当前Koopman算子用无监督方式拟合，预测误差≠分类性能
**方案**: 用在线分类性能作为反馈，修正K_t估计，使其更有利于判别

---

## 与现有方法的对比

### 当前DCA-BGF
```
协方差空间 → 静态RA对齐 → CSP特征 → 分类
问题: 对齐与动力学分离，gate决策逻辑不清晰
```

### 窗口级KCAR
```
协方差空间 → Koopman诊断 → 离散策略
问题: 诊断与对齐分离，未充分利用Koopman结构
```

### KSDA (新方案)
```
Koopman空间 → 动力学对齐 → 性能反馈 → 在线适应
优势: 统一框架，端到端优化，理论保证
```

---

## 预期优势

### 理论优势
1. **计算复杂度**: O(m²) vs O(n³)，快10-100倍
2. **在线学习**: 线性空间可证明regret bound
3. **动力学一致性**: 对齐保持谱结构

### 实践优势
1. **更高效**: 避免矩阵分解，纯线性运算
2. **更统一**: KCAR和对齐在同一空间
3. **更灵活**: 支持在线更新和元学习

---

## 实验路线图

### Phase 1: 基础验证 (1-2周)
**Exp-D.1**: 静态Koopman对齐
→ 验证Koopman空间对齐的基础可行性

### Phase 2: 动态适应 (2-3周)
**Exp-D.2**: 在线更新对齐矩阵
**Exp-D.3**: 性能加权的Koopman更新

### Phase 3: 风险感知 (1-2周)
**Exp-D.4**: KCAR驱动的动态对齐

### Phase 4: 元学习 (2-3周)
**Exp-D.5**: 端到端元学习版本

### Phase 5: 完整验证 (3-4周)
- 多数据集验证
- 完整消融研究
- 理论分析

---

## 成功标准

### 最小成功标准 (Phase 1)
- KSDA-static ≥ RA baseline (43.4%)
- 计算时间 < RA

### 中等成功标准 (Phase 2-3)
- KSDA-dynamic > KSDA-static + 2%
- 超越窗口级KCAR策略

### 完全成功标准 (Phase 4-5)
- KSDA-meta > fixed w=1.0 + 3%
- 在2+个数据集上验证
- 理论regret bound证明

---

## 风险与应对

### 风险1: Koopman空间对齐可能不如协方差空间
**应对**: 如果Exp-D.1失败，分析是lifting问题还是本质问题

### 风险2: 性能反馈噪声太大
**应对**: 滑窗平均，高置信度更新，正则化

### 风险3: 元学习样本不足 (只有9个被试)
**应对**: 数据增强，多数据集预训练，简化目标

---

## 论文定位 (如果成功)

### Title
Koopman-Space Dynamic Alignment for Cross-Subject Motor Imagery BCI

### 核心贡献
1. 首次在Koopman空间做跨被试对齐
2. 性能驱动的Koopman算子在线修正
3. 统一的风险感知动态适应框架

### 目标会议
- **首选**: NeurIPS 2026 (理论创新)
- **备选**: ICML 2027 (Koopman理论)
- **保底**: AAAI 2027 (应用价值)

---

## 文档结构

- `00-overview.md`: 本文档，总览
- `01-exp-d1-static-alignment.md`: Phase 1实验设计
- `02-exp-d2-online-update.md`: Phase 2实验设计
- `03-exp-d3-performance-feedback.md`: Phase 3实验设计
- `04-exp-d4-risk-aware.md`: Phase 4实验设计
- `05-exp-d5-meta-learning.md`: Phase 5实验设计
- `06-theory.md`: 理论分析
- `07-implementation-notes.md`: 实现细节
- `08-results-tracking.md`: 结果追踪

---

## 当前状态

**日期**: 2026-03-08
**阶段**: 规划完成，准备开始Exp-D.1
**决策**: 基于Exp-B.1的负结果，转向Koopman空间对齐方向
