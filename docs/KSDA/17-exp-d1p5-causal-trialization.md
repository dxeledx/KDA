# Exp-D1.5：causal trialization

**定位**: KSDA v3.1 的桥接层  
**目的**: 用更接近在线的 trial-level 因果标签替代直接的 window oracle 评估

---

## 老师信号

第一版老师锁死为：
- `window=16`
- **上一完整窗口的 oracle action**
- 第一个窗口默认 `A0_identity`

这样得到严格因果的 piecewise-constant trial labels。

---

## 比较对象

- `best single action`
- `causal-trialized oracle`
- `non-causal window oracle`（仅作上界参考）

---

## 通过标准

`D1.5` 通过必须同时满足：
- `causal-trialized oracle` 至少优于 `best single action +0.5%`
- `wins >= 6/9`
- 最常见动作占比 `< 80%`
- 单被试贡献不超过总增益的 `40%`
- 与原 window oracle 的动作一致率 `>= 70%`

若不通过，则不进入 `D2 / D3`
