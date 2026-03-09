# RBID / K-RBID 指标定义与当前结论

**日期**: 2026-03-09

## 全局指标：RBID

- `s_rep_ij`: 源 `i` 与目标 `j` 的表征相似度  
  - Stage 1 用 `CKA`
  - KSDA 阶段切到 Koopman 特征空间中的表征相似度
- `s_beh_ij`: 源 `i` 对目标 `j` 的行为一致性 / 可迁移性  
  - 主版本固定用 `transfer accuracy`
- 固定目标 `j` 后，在所有 `i≠j` 上做 rank normalization：
  - `ŝ_rep_ij`
  - `ŝ_beh_ij`
- 定义：
  - `RBID = mean |ŝ_rep_ij - ŝ_beh_ij|`
  - `RBID+ = mean [ŝ_rep_ij - ŝ_beh_ij]_+`
  - `RBID- = mean [ŝ_beh_ij - ŝ_rep_ij]_+`
  - `Tail-RBID = top 25% mismatch 平均值`

## 局部指标：K-RBID

- 在当前窗口 `t`、候选动作 `a` 下：
  - 几何分数：`g_t(a)`
  - 行为分数：`p_t(a)`
- 第一版主定义：
  - `g_t(a) = - residual_t(a)`，使用动作条件化的 Koopman transition residual
  - `p_t(a) = pseudo-oracle gain_t(a)`
- 在动作集合内做 rank normalization 后：
  - `d_t(a) = |ranknorm_a(g_t(a)) - ranknorm_a(p_t(a))|`
  - `K-RBID_t = mean_a d_t(a)`

## 当前结论

- `RBID` 已经能作为“现象存在且现有方法仍存在该问题”的更强指标，优于只看 Pearson。
- `K-RBID` 当前第一版只做诊断层：
  - 它与 pseudo-oracle gain 呈负相关
  - 高 `K-RBID` 的窗口平均 gain 更低
- 当前还**不直接**把 `K-RBID` 接入 `D2 / D3`
- 只有当局部 `K-RBID` 的诊断价值进一步稳定后，才把它升级成方法驱动量
