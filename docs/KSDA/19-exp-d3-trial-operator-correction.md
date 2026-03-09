# Exp-D3：trial-level 保守算子修正

**定位**: KSDA v3.1 的第二层想法验证  
**目的**: 在 trial-level 因果约束下，用线性 scalar proxy 驱动保守的低秩算子修正

---

## 修正形式

`K_tilde = K_hat + α_t U V^T`

其中：
- `U, V` 来自 `K_t0 - K_s` 的 top-1 奇异方向
- `α_t = clip(γ * g_t, -0.1, 0.1)`
- `γ ∈ {0.05, 0.1}`

更新频率锁死为：
- 每 `8` 个 trial 更新一次
- 每次只看最近 `16` 个 trial 的 causal 特征

---

## 通过标准

`D3` 通过必须同时满足：
- residual 改善
- accuracy 同时提升 `+0.5%`
- 无明显数值发散
- 最差被试退化不超过 `-2%`
