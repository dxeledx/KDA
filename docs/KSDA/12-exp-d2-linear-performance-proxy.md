# Exp-D2：线性性能指标预测动作收益

**定位**: KSDA v3 的信号层验证  
**目的**: 不直接修算子，先验证线性性能指标是否能预测“当前窗口更适合哪种专家动作”

---

## 核心特征

窗口级特征锁死为：

`m_t = [r_src, r_tgt, Δr, margin, drift, stability]`

其中：
- `r_src`: 相对 `K_s` 的 transition residual
- `r_tgt`: 相对当前固定 `K_t` 的 transition residual
- `Δr = r_tgt - r_src`
- `margin`: source-trained LDA 在当前 window 的平均分类间隔
- `drift`: window 在 `z/ψ` 空间中的漂移量
- `stability`: 预测一致性或状态平滑度

---

## 学习目标

不是直接学最终 accuracy，而是学：

> 当前 window 更适合哪种专家动作

训练标签来自 `D1` 的 pseudo-target oracle。

---

## 比较对象

- `best single expert`
- `oracle expert`
- `linear-proxy selector`
- `geom-only selector`

---

## 通过标准

`D2` 通过必须同时满足：

1. `linear-proxy selector > geom-only selector`
2. `linear-proxy selector >= best single expert`
3. 对 oracle 的动作一致率显著高于随机

若不满足，则不进入 `D3`
