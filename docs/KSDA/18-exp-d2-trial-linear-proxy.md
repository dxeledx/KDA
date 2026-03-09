# Exp-D2：trial-level 线性动作收益 proxy

**定位**: KSDA v3.1 的信号学习层  
**目的**: 用严格因果的逐 trial 标签训练线性 selector，验证动作收益是否可被线性 proxy 预测

---

## 标签来源

训练标签只来自 `D1.5` 的因果逐 trial 动作标签。

---

## 特征

trial-level causal trailing-16 特征：

`m_t = [r_src, r_tgt, Δr, margin, drift, stability]`

全部只使用 `≤ t` 的信息。

---

## 比较对象

- `best single action`
- `causal-trialized oracle`
- `linear-proxy selector`
- `geom-only selector`

---

## 通过标准

`D2` 通过必须同时满足：
- `linear-proxy selector > geom-only selector +0.5%`
- `linear-proxy selector >= best single action`
- 对 `causal-trialized oracle` 的动作一致率 `>= 45%`
- 且高于随机 `+15%`
