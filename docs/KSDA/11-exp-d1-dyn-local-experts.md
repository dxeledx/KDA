# Exp-D1：动态局部专家集 benchmark

**定位**: KSDA v3 的动态动作空间验证  
**目的**: 验证状态能否驱动多个 Koopman 局部线性专家之间的切换

---

## 核心问题

这一步不再问“单最佳静态器是谁”，而是问：

> 在同一个 Koopman 表征上，多个局部线性动作之间是否存在互补性？

如果 `oracle-expert` 也塌缩到单专家，那么当前问题就不在“怎么动态选”，而在“动作空间还不够有意义”。

---

## 专家集

- `E0`: identity
- `E1`: conservative mean shift
- `E2`: diagonal scaling
- `E3`: low-rank shrinkage affine
- `E4`: supervised subspace aligner

全部固定在同一个 Koopman 表征上：
- `PCA=16`
- `quadratic lifting`

---

## 评估方式

第一版只做：
- 单专家全局固定使用
- `window-level oracle expert`

不做专家间插值，不学 controller。

---

## 通过标准

`D1` 通过必须同时满足：

1. `oracle-expert` 明显优于任一单专家
2. 最优专家选择不塌缩
3. 提升不能只靠单个异常被试撑起

若不满足，则不进入 `D2`

---

## 输出

- `single_expert_summary.csv`
- `oracle_expert_summary.csv`
- `expert_action_histogram.csv`
- `expert_switch_stats.csv`
- `details/subject_Axx.npz`
- `summary.json`
