# KDA / DCA-BGF Research Repo

这个仓库记录了一个跨被试运动想象（Motor Imagery, MI）脑机接口研究项目的完整迭代过程：

- **阶段一**：复现静态 baseline，验证“表征-行为不一致”现象
- **阶段二**：实现并检验 `DCA-BGF`（Dynamic Conditional Alignment with Behavior-Guided Feedback）
- **阶段三**：从“连续动态控制”转向 **RA-first risk-aware selective adaptation**
- **当前下一步**：规划 `KSDA`（Koopman-Space Dynamic Alignment）方向

换句话说，这个仓库不是单一算法的最终版代码，而是一个**研究工作台**：包含代码、实验结果、阶段文档、失败分析和下一阶段的研究设计。

---

## 当前状态（2026-03-08）

- **数据集**：`BNCI2014001`（BCI Competition IV 2a，9 个被试，4 类 MI）
- **已完成**：
  - Stage 1 baseline 复现与现象验证
  - Stage 2 DCA-BGF MVP / full / ablation / rescue
  - Stage 3 KCAR 风险诊断、RA-first policy benchmark、trial-level 动态门控实验
- **当前结论**：
  - `RA` 仍然是当前设定下最稳健的默认策略
  - “连续动态 gating + 行为反馈”这条主线**没有稳定超过 RA**
  - 风险感知的**选择性偏离 RA**有价值，`KCAR` policy 在当前结果中出现了小幅正增益
- **下一步**：
  - 继续推进 `docs/KSDA/` 中定义的 Koopman 空间新路线
  - 当前仓库里 **KSDA 还是规划文档，没有对应实现代码**

---

## 一句话结论

这个项目的研究主线已经从：

> “动态连续对齐是否能优于静态 RA？”

演进为：

> “在 RA 作为强默认策略时，能否用风险信号只在合适窗口偏离 RA，从而减少负迁移？”

---

## 项目迭代路径

| 日期 | 阶段 | 主要产物 | 结论 |
|------|------|----------|------|
| `2026-03-05` | Stage 1 | `experiments/baseline_csp_lda.py`、`experiments/baseline_ea.py`、`experiments/baseline_ra.py`、`experiments/phenomenon_verification.py`、`docs/stage1/` | 复现 no-align / EA / RA；确认存在表征-行为不一致现象 |
| `2026-03-06` | Stage 2 | `experiments/dca_bgf_mvp.py`、`experiments/dca_bgf_full.py`、`experiments/ablation_study.py`、`experiments/stage2_rescue.py`、`docs/stage2/` | DCA-BGF 当前实现未稳定超过 RA，项目主线开始 pivot |
| `2026-03-07` | Stage 3 | `experiments/kcar_risk_diagnostic.py`、`experiments/kcar_safe_policy.py`、`experiments/trial_dynamic_gate_exp_a.py`、`experiments/trial_dynamic_gate_exp_b.py`、`experiments/trial_dynamic_gate_exp_b1.py`、`docs/stage3/` | 转向 RA-first risk-aware selective adaptation；KCAR policy 出现小幅收益 |
| `2026-03-08` | Next | `docs/KSDA/` | 基于 Stage 3 的结论，规划 Koopman-Space Dynamic Alignment（KSDA） |

如果你想按时间顺序理解整个项目，最推荐的文档路径是：

1. `docs/research-dynamic-bci-alignment-2026-03/research-proposal.md`
2. `docs/stage1/README.md`
3. `docs/stage2/07-failure-analysis-and-stage3-pivot.md`
4. `docs/stage3/00-overview.md`
5. `docs/stage3/01-kcar-risk-diagnostic.md`
6. `docs/stage3/02-ra-first-policy-benchmark.md`
7. `docs/KSDA/00-overview.md`

---

## 当前最重要的结果

以下数字都来自仓库里已经提交的结果文件，评价设置以 **LOSO cross-subject accuracy** 为主。

### Stage 1：静态 baseline

| 方法 | Mean Accuracy | 结果文件 |
|------|---------------|----------|
| No Alignment | `0.3816` | `results/baselines/noalign/summary.json` |
| EA | `0.4236` | `results/baselines/ea/summary.json` |
| RA | `0.4275` | `results/baselines/ra/summary.json` |

### 表征-行为相关性

| 方法 | Pearson r | 结果文件 |
|------|-----------|----------|
| No Alignment | `0.3919` | `results/baselines/noalign/correlation.json` |
| EA | `0.6467` | `results/baselines/ea/correlation.json` |
| RA | `0.6410` | `results/baselines/ra/correlation.json` |

这也是项目最早的研究动机来源：**单纯看表征相似度并不足以解释行为层面的跨被试迁移表现**。

### Stage 2：DCA-BGF 主线没有超过 RA

| 方法 | Mean Accuracy | 相对 RA |
|------|---------------|---------|
| RA | `0.4275` | `0.0000` |
| DCA-BGF (MVP) | `0.4248` | `-0.0027` |
| DCA-BGF (full) | `0.4097` | `-0.0177` |
| Rescue 中最佳 adaptive | `0.4213` | `-0.0062` |

对应文件：

- `results/stage2/mvp/summary.json`
- `results/stage2/full/summary.json`
- `results/stage2/rescue/2026-03-06-stage2-rescue-v3/best_candidate.json`
- `docs/stage2/07-failure-analysis-and-stage3-pivot.md`

这一步的关键结论不是“问题不存在”，而是：

- **问题存在**：负迁移控制仍然是重要研究问题
- **当前解法不成立**：连续动态权重控制器没有学到比“始终做 RA”更好的决策边界

### Stage 3：RA-first 风险感知策略开始出现信号

#### KCAR 风险诊断

- `mean_auroc = 0.6877`
- `mean_auprc = 0.7201`
- `mean_spearman = 0.2727`

见 `results/stage3/kcar_diagnostic/2026-03-07-kcar-diagnostic-v2/summary.json`。

#### RA-first policy benchmark

当前最好的策略结果来自：

- `policy = kcar`
- `setting = near_causal`
- `coverage = 0.2`
- `mean_accuracy = 0.4325`
- `mean_delta_vs_ra = +0.0050`

见 `results/stage3/kcar_policy/2026-03-07-kcar-policy-v1/policy_summary.csv`。

这说明：

- **“完全替代 RA” 还做不到**
- 但 **“在少量窗口里有条件地偏离 RA”** 已经出现了可用信号

#### Trial-level 连续动态实验（Stage 3 另一条线）

| 方法 | Mean Accuracy | 文件 |
|------|---------------|------|
| Exp-A dynamic | `0.4147` | `results/stage3/trial_dynamic_gate_exp_a/2026-03-07-trial-dynamic-exp-a/summary.json` |
| Exp-B dynamic | `0.4136` | `results/stage3/trial_dynamic_gate_exp_b/2026-03-07-trial-dynamic-exp-b/summary.json` |
| Exp-B.1 dynamic | `0.4074` | `results/stage3/trial_dynamic_gate_exp_b1/2026-03-07-trial-dynamic-exp-b1/summary.json` |
| Fixed `w=1.0` | `0.4340` | 同上各 `summary.json` |

这进一步强化了同一个结论：**复杂的连续 gate 依旧没有打赢强默认策略 `RA / fixed w=1.0`。**

---

## 仓库结构

```text
.
├── configs/        # 数据、实验、模型配置
├── docs/           # 阶段文档、失败分析、研究备忘录、后续方案
├── experiments/    # 各阶段实验入口脚本
├── results/        # 已提交的实验产物、摘要、图表
├── scripts/        # 预处理和验证脚本
├── src/            # 数据、对齐、特征、模型、评估核心实现
└── tests/          # Stage 1 ~ Stage 3 相关测试
```

### 代码模块说明

- `src/data/`：MOABB 数据加载、缓存、预处理
- `src/features/`：协方差与 CSP 特征
- `src/alignment/`：EA / RA / conditional alignment / DCA-BGF / feedback
- `src/models/`：分类器封装（当前主要是 LDA）
- `src/evaluation/`：LOSO、pairwise transfer、KCAR 分析、policy benchmark、可视化
- `src/utils/`：配置、日志、上下文特征等工具

---

## 环境与安装

### Python 版本

- `python >= 3.8`
- 实际开发时更推荐 `python 3.9` 或 `python 3.10`

### 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> 当前仓库更适合作为**从仓库根目录直接运行的脚本型研究代码库**使用；  
> 大多数实验脚本会在启动时把项目根目录加入 `sys.path`，并直接导入 `src.*` 模块。

### 依赖说明

核心依赖见 `requirements.txt`，包括：

- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `mne`, `pyriemann`, `moabb`
- `matplotlib`, `seaborn`
- `pytest`, `pytest-cov`

---

## 数据准备

项目当前默认使用 `configs/data_config.yaml` 中配置的：

- 数据集：`BNCI2014001`
- 被试：`1-9`
- 通道：22 通道
- 时间窗：`3.0s - 6.0s`
- 频段：`8-30 Hz`

### 首次运行说明

- 原始数据文件和处理后的缓存**没有提交到仓库**
- 第一次运行会通过 `MOABB` 自动获取数据，并将处理结果缓存到 `data/processed`
- 这些缓存路径已经被 `.gitignore` 排除，不会污染版本库

### 预处理缓存

```bash
python3 scripts/preprocess_all.py
```

---

## 如何复现实验

### 1. Stage 1：baseline 与现象验证

```bash
python3 experiments/baseline_csp_lda.py
python3 experiments/baseline_ea.py
python3 experiments/baseline_ra.py
python3 experiments/phenomenon_verification.py
```

输出目录主要包括：

- `results/baselines/noalign`
- `results/baselines/ea`
- `results/baselines/ra`
- `results/figures`

### 2. Stage 2：DCA-BGF 主线

```bash
python3 experiments/dca_bgf_mvp.py
python3 experiments/dca_bgf_full.py
python3 experiments/ablation_study.py
python3 experiments/rep_behavior_stage2.py
python3 experiments/stage2_rescue.py --run-name latest
```

主要输出：

- `results/stage2/mvp`
- `results/stage2/full`
- `results/stage2/ablation`
- `results/stage2/rep_behavior`
- `results/stage2/rescue/<run-name>`

### 3. Stage 3：风险诊断与 RA-first policy

```bash
python3 experiments/kcar_risk_diagnostic.py --run-name latest
python3 experiments/kcar_safe_policy.py \
  --diagnostic-dir results/stage3/kcar_diagnostic/latest \
  --run-name latest \
  --force
```

主要输出：

- `results/stage3/kcar_diagnostic/<run-name>`
- `results/stage3/kcar_policy/<run-name>`

### 4. Stage 3：trial-level 动态 gate 分析

```bash
python3 experiments/trial_dynamic_gate_exp_a.py --run-name latest
python3 experiments/trial_dynamic_gate_exp_b.py --run-name latest
python3 experiments/trial_dynamic_gate_exp_b1.py --run-name latest
```

主要输出：

- `results/stage3/trial_dynamic_gate_exp_a/<run-name>`
- `results/stage3/trial_dynamic_gate_exp_b/<run-name>`
- `results/stage3/trial_dynamic_gate_exp_b1/<run-name>`

> 建议不要长期使用默认的 `latest` 作为正式实验名。  
> 想保留历史实验时，更推荐显式传入带日期的 `--run-name`。

---

## 验证与测试

### 基础验证脚本

```bash
python3 scripts/validate_stage1.py
```

### 测试

```bash
python3 -m pytest tests -q
```

测试覆盖了：

- 对齐模块
- CSP / CKA
- Stage 2 分析与反馈逻辑
- Stage 3 trial dynamic 实验的关键路径

---

## 推荐阅读路径

如果你第一次进入这个仓库，建议按下面顺序阅读：

### 先理解研究为什么转向

1. `docs/research-dynamic-bci-alignment-2026-03/research-proposal.md`
2. `docs/stage1/04-phenomenon-verification.md`
3. `docs/stage2/07-failure-analysis-and-stage3-pivot.md`

### 再看当前主线

4. `docs/stage3/00-overview.md`
5. `docs/stage3/01-kcar-risk-diagnostic.md`
6. `docs/stage3/02-ra-first-policy-benchmark.md`
7. `docs/stage3/08-exp-b1-results-memo.md`

### 最后看下一阶段方向

8. `docs/KSDA/00-overview.md`
9. `docs/KSDA/01-exp-d1-static-alignment.md`
10. `docs/KSDA/02-exp-d2-online-update.md`

---

## 仓库里已经有哪些结果

这个仓库已经提交了大量**可直接查看的实验结果**，包括：

- baseline 摘要 CSV / JSON
- Stage 2 rescue 扫描结果
- Stage 3 KCAR 诊断结果
- policy benchmark 汇总表
- 各种 PDF 图表

如果你只是想理解项目状态，**不一定需要重新跑实验**，可以先直接看：

- `results/baselines/`
- `results/stage2/`
- `results/stage3/`

---

## 当前研究判断

基于目前仓库中的结果，我对这个项目的状态判断是：

1. **Stage 1 是成功的**  
   baseline 与现象验证为后续工作提供了坚实起点。

2. **Stage 2 是“有价值的负结果”**  
   DCA-BGF 当前实现没有超过 RA，但它明确排除了“继续细调连续 gate 就能赢”的幻想。

3. **Stage 3 是主线重构阶段**  
   项目不再试图证明“连续动态控制一定更优”，而是转向“默认 RA、少量偏离”的安全自适应叙事。

4. **KSDA 是下一步假设，不是现成成果**  
   `docs/KSDA/` 很重要，但请注意它目前还是研究设计，不是已经跑完的实证结果。

---

## 许可证

当前仓库**尚未提供单独的 `LICENSE` 文件**。  
如果你计划复用代码或结果，建议先补充明确授权条款。
