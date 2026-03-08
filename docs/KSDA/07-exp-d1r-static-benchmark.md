# Exp-D1-R：静态 Koopman 对齐器重做 benchmark

**定位**: KSDA 新主线的第一步  
**目的**: 先把静态问题解耦，回答“Koopman 表征是否值得继续”和“当前失败到底来自表征还是来自静态对齐器 `A`”

---

## 1. 研究问题

`D1-R` 只回答三件事：

1. **Koopman 表征本身是否有基础可用性**
2. **在固定表征下，哪类静态对齐器更适合 Koopman 空间**
3. **某个方法的失败到底是“方法不好”，还是“数值没站稳”**

---

## 2. 固定 pipeline

对每个 trial：

```python
z_t = PCA(tangent_project(C_t, M_s))
ψ_t = lift(z_t)
ψ'_t = align_A(ψ_t)
y_pred = LDA(ψ'_t)
```

固定不变项：
- 数据集：`BNCI2014001`
- 协议：`LOSO`
- 预处理：与 `stage2/stage3` 保持一致
- 分类器：`LDA`

---

## 3. 三层实验结构

### 3.1 表征可用性 benchmark

只比较：
- `RA + CSP + LDA`
- `NoAlign + CSP + LDA`
- `Koopman ψ → LDA`

表征 sweep 固定为：
- `PCA rank ∈ {16, 32, 64}`
- `lifting ∈ {quadratic, quadratic+cubic}`

这里不加任何新的对齐器。

### 3.2 静态对齐器 benchmark

在固定最佳 `ψ*` 上，只换 `A`：

- `A0`: no-align
- `A1`: LDA-style / supervised projection
- `A2`: CSP-style generalized eigen
- `A3`: regularized linear `A`

`legacy-affine` 只作为冻结参考，不进入主候选选择。

### 3.3 稳定性 sweep

只围绕 `A1/A2/A3` 做：
- `k ∈ {8, 16, 32}`
- `λ ∈ {1e-4, 1e-3, 1e-2}`
- `aligned feature normalization ∈ {off, on}`

目标不是刷最优，而是避免误把数值问题当成方法失败。

---

## 4. 结果与推进门槛

输出固定为：
- `representation_benchmark.csv`
- `aligner_benchmark.csv`
- `stability_sweep.csv`
- `best_static.json`
- `summary.json`
- `details/subject_Axx.npz`

`best-static` 的选择顺序锁死为：
1. `mean accuracy`
2. `wins vs A0`
3. `无 fold 崩坏`
4. `更快/更简单`

只有满足以下三条，才进入 `D1+-R`：
- `best-static >= 42.0%`
- 无 fold 崩坏
- 相比 `A0`，类间/类内距离比有一致改善

否则不进入动态信号验证。
