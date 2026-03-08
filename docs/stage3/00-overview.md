# 阶段三总览：KCAR 驱动的 RA-first Safe Adaptation

## 目标

阶段三不再尝试证明“连续动态 gating 必然优于 RA”。  
新的目标是：

1. 保留 `RA + CSP + LDA/SVM` 作为安全基座；
2. 用 `KCAR` 判断**何时偏离 RA 可能更好**；
3. 先验证 `KCAR` 是否是有效的**负迁移风险诊断器**；
4. 只有在风险诊断通过后，才进入 `RA-first` 的窗口级二动作策略。

---

## 两阶段 Gate

### Phase A：KCAR 风险诊断

主实验入口：

- `experiments/kcar_risk_diagnostic.py`

主输出目录：

- `results/stage3/kcar_diagnostic/<run_name>/`

主问题：

> `KCAR` 能否识别“偏离 RA 更好”的窗口，并且优于简单 heuristic？

主验收标准：

- `mean AUROC >= 0.60`
- `KCAR` 在至少 `6/9` 个被试上优于任一单一 heuristic 的 AUROC
- `Spearman(ρ, delta_dev_vs_ra) > 0`

若不满足，阶段三停止在 failure-analysis，不进入策略实验。

### Phase B：RA-first Safe Policy

策略实验入口：

- `experiments/kcar_safe_policy.py`

主输出目录：

- `results/stage3/kcar_policy/<run_name>/`

策略形式：

- `stay_with_ra (w=1.0)`
- `use_partial_alignment (w=0.5)`

决策规则：

- 若 `ρ_window > τ_risk`，则使用 `w=0.5`
- 否则维持 `w=1.0`

主验收标准：

- 平均准确率不低于 `RA`
- `negative-transfer subject count` 不高于 `RA-first`
- `worst-subject delta` 不差于当前 `adaptive/best`
- 在相同 deviation coverage 下，`KCAR policy` 要与 `d_tgt policy`、`sigma_recent policy` 做强对照
- retrospective 与 near-causal 必须并排报告

---

## 固定实验协议

- 数据集：`BNCI2014001`
- 主协议：`LOSO`
- 预处理、协方差估计、CSP、分类器与 `stage2 rescue` 保持一致
- 第一版只做**窗口级 retrospective diagnosis**
- 不做伪标签更新、不做在线 feedback 规则、不做 trial 级控制

---

## 论文叙事

阶段三的论文主线固定为：

> **RA-first safe adaptation**

其中 `KCAR` 的角色不是主对齐器，而是：

> **诊断 source-driven alignment 在当前窗口是否存在高负迁移风险**

因此论文问题从“如何连续调对齐强度”改成：

> **何时应该坚持强默认策略，何时允许保守地偏离它？**

---

## 核心文件

- 方法与诊断定义：`docs/stage3/01-kcar-risk-diagnostic.md`
- 窗口级结果备忘录：`docs/stage3/03-kcar-window-policy-results-memo.md`
- Trial-level Exp-A 结果备忘录：`docs/stage3/05-exp-a-results-memo.md`
- Trial-level Exp-B 结果备忘录：`docs/stage3/06-exp-b-results-memo.md`
- Exp-B.1 设计说明：`docs/stage3/07-exp-b1-normalized-plan.md`
- Exp-B.1 结果备忘录：`docs/stage3/08-exp-b1-results-memo.md`
- 阶段二失败分析：`docs/stage2/07-failure-analysis-and-stage3-pivot.md`

---

## 当前建议执行顺序

1. `KCAR` 窗口级诊断与 `RA-first policy` benchmark 已完成
2. 当前结果记入 `03-kcar-window-policy-results-memo.md`
3. 下一步主实验切换到 **Exp-A：trial-level 动态 gate 最小版**
4. `Exp-A` 的结果记录在 `05-exp-a-results-memo.md`
5. `Exp-B` 的结果记录在 `06-exp-b-results-memo.md`
6. 在进入 `rho_window` 前，优先做 `Exp-B.1`（归一化修正版）
7. `Exp-B.1` 的设计与结果分别记录在 `07` / `08`
8. `Exp-B.1` 说明“不是单纯尺度问题”，因此是否继续进入 `Exp-C` 需要重新判断
9. 只有当 trial-level `w_t` 本身站住后，才进入 `rho_window` 注入与 feedback
10. 详细顺序见 `04-trial-level-roadmap.md`
