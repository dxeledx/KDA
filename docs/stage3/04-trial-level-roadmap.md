# 阶段三 Trial-Level 路线图（占位）

## 下一步实验顺序

### Exp-A：trial-level 动态 gate 最小版

- 基座：`RA + CSP + LDA`
- 输入：`d_tgt only`
- 输出：连续 `w_t`
- 对照：`fixed w=1.0 / 0.5 / 0.0`

目标：

- 看 trial-level 动态 `w_t` 是否本身成立
- 看 `w_t` 是否有结构，而不是塌成常数
- 看它是否开始接近或超过 fixed baseline

### Exp-B：trial-level 动态 gate 几何版

- 输入：`d_src + d_tgt + sigma_recent`
- 其余不变

目标：

- 判断 richer geometry 是否优于 `d_tgt only`

### Exp-C：trial-level 动态 gate + 窗口风险慢变量

- 输入：`d_src + d_tgt + sigma_recent + rho_window`

目标：

- 验证窗口级 risk signal 是否能帮助 trial-level 动态 gate

### Exp-D：trial-level 动态 gate + rule1 feedback

- 在 `Exp-B` 或 `Exp-C` 最优版上
- 只加最简单的不确定性反馈

目标：

- 看 session 后期稳定性是否进一步提升
