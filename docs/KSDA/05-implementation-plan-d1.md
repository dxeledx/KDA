# KSDA Phase 1 实现计划

## 范围

当前只实现：

1. `Exp-D.1`: 静态 Koopman 空间对齐
2. `Exp-D.1+`: 固定 `A / K_s / K_t` 的 KCAR gate 验证

不实现：

- `D.2` 在线更新 `A_t`
- `D.3` 性能反馈 `K_t`
- `D.4` 风险感知完整版本

---

## D.1 关键定义

- `ψ_t = lift_quadratic(PCA(tangent_project(C_t, M_s)))`
- `A` 是 Koopman 特征空间里的 source→target 二阶统计仿射对齐
- `LDA` 训练在 source `ψ_source`
- target test 用对齐后的 `ψ'_t`

---

## D.1+ 关键定义

- `A` 固定为 D.1 学到的静态对齐
- `K_s` 固定为 source-train 拟合的全局 Koopman 算子
- `K_t` 固定为 target-train 一次性拟合的目标 Koopman 算子
- `ρ_t` 用最近最多 32 个历史 trial 计算
- `w_t = sigmoid(1.0 - 2.0 * ρ_t)`

---

## 止损规则

- 如果 `KSDA-static < RA - 1%`，停止 D.1+
- 如果 `KSDA-kcar-gate ≈ KSDA-geometric-gate`，不进入 D.2 / D.3
