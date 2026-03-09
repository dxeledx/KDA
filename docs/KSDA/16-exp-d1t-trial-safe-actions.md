# Exp-D1-T：trial-safe 二维动作集

**定位**: KSDA v3.1 的第一步  
**目的**: 先按 `trial-level / online / 只看过去信息` 的约束重构动作集

---

## 核心思想

动作不再只描述“用什么方法对齐”，而是定义为：

`动作 = 对齐端点 × 对齐强度`

统一写成：

`ψ'_t = ψ_t + alpha * (T_endpoint,t(ψ_t) - ψ_t)`

---

## 动作集

端点：
- `P1`: history mean shift
- `P2`: history diagonal scaling
- `P3`: train-fixed low-rank shrinkage affine
- `P4`: source-trained supervised subspace

强度：
- `alpha ∈ {0.0, 0.33, 0.67, 1.0}`

去重后得到 13 个动作：
- `A0_identity`
- 每个端点的 `alpha ∈ {0.33, 0.67, 1.0}`

---

## 评估方式

第一版只做：
- 每个动作整段 target test 的 **trial-level fixed action benchmark**

不做窗口 oracle，不做 selector。

---

## 通过标准

`D1-T` 通过必须满足：
- 非 `identity` 动作里，不会有至少 2 个同时与 `A0` 的预测重合率 `> 95%`
- 非 `identity` 动作里，不会有至少 2 个平均变换幅度接近 0
- top-3 动作不全部来自同一端点家族

若失败，则先重构动作集，不进入 `D1.5`
