# Exp-D1+-R：只在最优静态器上重测动态信号

**定位**: KSDA 新主线的第二步  
**目的**: 只在可靠静态基座上回答“`KCAR` 是否真的优于几何信号”

---

## 1. 前置条件

`D1+-R` 不是默认下一步，只有 `D1-R` 通过以下门槛才执行：
- `best-static >= 42.0%`
- 无 fold 崩坏
- 对齐质量相对 `A0` 有一致改善

---

## 2. 固定对象

`D1+-R` 只用 `D1-R` 选出来的：
- 最佳表征 `ψ*`
- 最佳静态器 `A*`

并固定：
- `K_s`: source-train 拟合
- `K_t`: target-train/calibration 一次性拟合，测试时不更新

不再重新混入多种 `A`。

---

## 3. 正式对照

只跑四个方法：
- `best-static`
- `geom-gate`
- `kcar-gate`
- `window-kcar`

其中：
- `geom-gate`: 延续 Koopman `z` 空间里的几何 gate
- `kcar-gate`: 延续连续 `ρ_t → w_t` gate
- `window-kcar`: 保守 2 动作版本

`window-kcar` 锁死为：
- 非重叠 `32`-trial 窗
- 第一个窗口固定 `w=1.0`
- 从第二个窗口起，使用上一个完整窗口的 `rho_window`
- `rho_window > 0 → w=0.5`，否则 `w=1.0`

---

## 4. 成功标准

`D1+-R` 分两层判定：

### signal success

`kcar-gate > geom-gate`

至少要求：
- `mean accuracy` 高 `0.5%+`
- 且在多数被试上不是全面落后

### method success

`kcar-gate >= best-static`

只有 **signal success + method success** 同时成立，才说明：
- `KCAR` 不只是“有信息”
- 它还能在可靠静态基座上转成更好的动作

---

## 5. 对 D2/D3 的影响

只有同时满足以下三条，才允许继续 `D2 / D3`：

1. `best-static` 已经接近 `RA`
2. `kcar-gate` 明显优于 `geom-gate`
3. `kcar-gate` 至少不弱于 `best-static`

否则：
- 不继续 `D2 / D3`
- 根据结果改走窗口级策略，或把 Koopman 空间定位成静态分析 / 风险诊断工具
