# DCA-BGF: Dynamic Conditional Alignment with Behavior-Guided Feedback

**动态条件对齐与行为引导反馈的跨被试运动想象脑机接口方法**

---

## 📋 项目概述

本项目旨在解决跨被试运动想象（MI）脑机接口（BCI）中的**表征-行为不一致**问题，提出一种动态条件对齐方法，结合行为引导反馈机制，实现更好的跨被试泛化性能。

### 核心创新
1. **条件对齐网络**：根据当前上下文动态预测对齐参数
2. **行为引导反馈**：利用在线性能指标调整对齐策略
3. **闭环适应系统**：持续自我修正，适应within-session动态变化

### 目标
- **顶会投稿**：NeurIPS 2026, ICML 2026, ICLR 2027
- **时间线**：1-2个月完成实验和论文撰写

---

## 📂 项目结构

```
DCA-BGF/
├── docs/                          # 文档
│   └── stage1/                    # 阶段一：基线方法复现
│       ├── README.md              # 阶段一总览（从这里开始！）
│       ├── 00-overview.md
│       ├── 01-data-preparation.md
│       ├── 02-baselines-traditional.md
│       ├── 04-phenomenon-verification.md
│       ├── 06-debugging-guide.md
│       └── 07-code-organization.md
│
├── data/                          # 数据目录（待创建）
├── src/                           # 源代码（待创建）
├── experiments/                   # 实验脚本（待创建）
├── results/                       # 实验结果（待创建）
├── configs/                       # 配置文件（待创建）
└── README.md                      # 本文件
```

---

## 🚀 快速开始

### 1. 阅读文档
```bash
# 从阶段一开始
cd docs/stage1
open README.md  # 或用你喜欢的编辑器打开
```

### 2. 搭建环境
```bash
# 创建虚拟环境
conda create -n dca-bgf python=3.9
conda activate dca-bgf

# 安装依赖（待创建requirements.txt）
pip install numpy scipy scikit-learn mne pyriemann moabb matplotlib seaborn
```

### 3. 开始编码
按照 `docs/stage1/README.md` 的指引，逐步完成：
1. 数据准备
2. 基线方法复现
3. 现象验证

---

## 📅 研究路线图

### 阶段一：基线方法复现（2-4周）✅ 当前阶段
- [x] 文档已完成
- [ ] 数据准备
- [ ] CSP+LDA baseline
- [ ] EA/RA baseline
- [ ] 表征-行为不一致现象验证

### 阶段二：方法设计与实现（3-4周）
- [ ] 条件对齐网络设计
- [ ] 行为引导反馈机制
- [ ] MVP版本实现
- [ ] 初步实验验证

### 阶段三：实验验证（2-3周）
- [ ] 跨被试性能评估（LOSO）
- [ ] 消融研究
- [ ] 时间动态性分析
- [ ] 表征-行为分析

### 阶段四：理论分析（1-2周）
- [ ] 收敛性分析
- [ ] 泛化界推导
- [ ] 表征-行为关系形式化

### 阶段五：论文撰写（2-3周）
- [ ] Introduction & Related Work
- [ ] Method & Experiments
- [ ] 图表制作
- [ ] 投稿

---

## 🎯 研究目标

### 性能目标
| 方法 | Cross-Subject (LOSO) | 提升 |
|------|---------------------|------|
| CSP+LDA (baseline) | 40-50% | - |
| RA (最好的静态方法) | 60-70% | +15-20% |
| OTTA (最新在线方法) | 65-75% | +20-25% |
| **DCA-BGF (目标)** | **>75%** | **>30%** |

### 理论贡献
- 收敛性保证
- 泛化界
- 表征-行为关系的形式化

### 实验贡献
- 全面的消融研究
- 时间动态性分析
- 多数据集验证

---

## 📊 核心方法（简要）

### 算法框架
```
输入: 源域数据 D_s, 目标域数据 D_t

离线阶段:
1. 训练基础分类器 f_s
2. 训练条件对齐网络 g(x_t, c_t) → θ_t

在线阶段 (逐trial):
1. 计算上下文 c_t (几何距离、预测熵、置信度趋势)
2. 预测对齐参数 θ_t = g(x_t, c_t)
3. 对齐特征 x'_t = Align(x_t, θ_t)
4. 预测 y_t = f_s(x'_t)
5. 行为引导反馈: 根据性能指标调整对齐策略
```

详细设计见阶段二文档（待创建）。

---

## 📚 参考文献

### 核心论文
- **Euclidean Alignment**: He & Wu (2019) "Transfer Learning for Brain-Computer Interfaces"
- **Riemannian Alignment**: Zanini et al. (2018) "Transfer Learning: A Riemannian Geometry Framework"
- **OTTA**: Calibration-free online test-time adaptation (2023)
- **CKA**: Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"

### 数据集
- **BCI Competition IV Dataset 2a**: http://www.bbci.de/competition/iv/

### 工具库
- **MNE-Python**: https://mne.tools/
- **PyRiemann**: https://pyriemann.readthedocs.io/
- **MOABB**: https://moabb.neurotechx.com/

---

## 🤝 协作方式

### 代码Review流程
1. 完成一个阶段的代码
2. 运行验证脚本（见 `docs/stage1/06-debugging-guide.md`）
3. 呼叫AI助手进行review
4. 根据反馈修改
5. 进入下一阶段

### 文档更新
- 每个阶段完成后，更新对应的文档
- 记录遇到的问题和解决方案
- 补充实验结果和可视化

---

## 📞 联系方式

- **研究者**: [Your Name]
- **机构**: [Your Institution]
- **邮箱**: [Your Email]

---

## 📄 许可证

本项目用于学术研究。如果使用本项目的代码或方法，请引用相关论文（待发表）。

---

## 🙏 致谢

感谢以下资源和工具：
- BCI Competition IV 数据集
- MNE-Python, PyRiemann, MOABB 开源库
- Claude AI 提供的研究指导

---

**最后更新**: 2026-03-05

**当前状态**: 阶段一文档已完成，准备开始编码 🚀
