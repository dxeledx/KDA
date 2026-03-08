# Exp-D4：闭环动态对齐

**定位**: KSDA v3 的闭环版本  
**前提**: 只有 `D0 / D1 / D2 / D3` 全部通过才执行

---

## 组合内容

`D4` 只在前面证据链全部站住之后，组合三件事：

- Koopman 空间表征
- 动态专家选择
- 线性性能指标驱动的算子修正

---

## 最终对照

- `RA + CSP + LDA`
- `Koopman-noalign`
- `best static expert`
- `dynamic experts`
- `dynamic experts + operator correction`
