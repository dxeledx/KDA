# KCAR 风险诊断实验设计

## 1. 新问题定义

阶段三第一版不试图学习一个完整的在线控制器，而是先验证：

> `KCAR` 是否能作为“偏离 RA 的风险/机会信号”。

因此 `KCAR` 的目标不是直接提升最终 accuracy，而是预测：

- 哪些窗口里 `RA` 是安全默认；
- 哪些窗口里 `w=0` 或 `w=0.5` 可能比 `RA` 更好。

---

## 2. 状态定义

### 2.1 几何状态

对每个 trial 协方差 `C_t`，以源域均值 `M_s` 为参考，取 tangent-space 表示：

- 先计算 `log(M_s^{-1/2} C_t M_s^{-1/2})`
- 再向量化为对称矩阵向量
- 再用 `PCA(r=16)` 得到 `z_t`

第一版 `KCAR` 的动力学状态只用 `z_t`，不把 heuristic 直接拼进 Koopman 状态。

### 2.2 Heuristic 对照信号

窗口级同时记录：

- `d_src`
- `d_tgt`
- `sigma_recent`
- `entropy`
- `conf_max`

这些信号只用于比较，不进入第一版 `KCAR` 动力学建模。

---

## 3. Koopman / KCAR 定义

### 3.1 全局 source 算子

在所有 source train subjects 上，按被试内时序构造状态转移：

- 只在同一被试内部连边
- 不跨被试拼接转移

第一版固定：

- 不做类条件拆分
- 不做 learnable lifting
- lifting 直接用 `ψ(s) = [s, s⊙s, 1]`
- ridge 正则固定

### 3.2 局部 target 算子

对每个 target test window（长度 `m=32`）：

- 用该窗口内部状态拟合一个局部线性 Koopman 算子 `K_t`

这是**retrospective diagnosis**，不是在线控制声明。

### 3.3 KCAR 风险值

对窗口内每个转移，计算：

- `e_src`：source 全局算子解释误差
- `e_tgt`：local target 算子解释误差

定义：

`ρ = mean((e_src - e_tgt) / (e_src + e_tgt + eps))`

并显式裁剪/验证到 `[-1, 1]`。

解释固定为：

- `ρ > 0`：source 动力学解释更差，**偏离 RA 值得考虑**
- `ρ < 0`：source 动力学解释较好，**继续 RA 更安全**

---

## 4. 窗口标签定义

主标签不是人工标签，而是相对 `RA` 的 oracle 决策结果：

- 若 `max(acc_w0, acc_w05) - acc_ra >= 1/32`  
  → `deviation-beneficial`
- 若 `acc_ra - max(acc_w0, acc_w05) >= 1/32`  
  → `ra-safe`
- 否则  
  → `neutral`

其中：

- `w=0.0` 代表 raw/no extra alignment
- `w=0.5` 代表 partial alignment
- `w=1.0` 即 `RA`

`neutral` 不进入 AUROC/AUPRC，但保留用于 `Spearman(ρ, delta_dev_vs_ra)`。

---

## 5. 主输出

### 5.1 表格

`window_metrics.csv`

字段：

- `subject`
- `window_id`
- `acc_ra`
- `acc_w0`
- `acc_w05`
- `delta_dev_vs_ra`
- `rho_kcar_retro`
- `rho_kcar_causal`
- `d_src`
- `d_tgt_retro`
- `d_tgt_causal`
- `sigma_recent_retro`
- `sigma_recent_causal`
- `entropy`
- `conf_max`
- `label`
- `budget_rank_*`（按被试内排序的 budget 匹配字段）

`comparison.csv`

字段：

- `subject`
- `score`
- `auroc`
- `auprc`
- `spearman`
- `eligible_windows`

`summary.json`

字段：

- `mean_auroc`
- `mean_auprc`
- `mean_spearman`
- `subject_wins_vs_heuristics`
- `eligible_window_count`

补充产物：

- `details/subject_Axx.npz`
  - `y_true`
  - `y_pred_ra`
  - `y_pred_w05`
  - `window_id_by_trial`

这些 detail 文件供第二阶段 `RA-first` policy benchmark 重建 trial 级预测与 `macro-F1 / bACC`。

### 5.2 图

- `rho_vs_window_accuracy.pdf`
- `rho_vs_delta_scatter.pdf`
- `rho_distribution_by_label.pdf`

---

## 6. 通过与失败解释

### 通过

若 `KCAR`：

- AUROC 过线
- 在多数被试上优于 heuristic
- 与 `delta_dev_vs_ra` 正相关

则说明：

> `KCAR` 有资格进入第二阶段 `RA-first` 策略。

### 失败

若 `KCAR` 未过线，则应优先接受以下结论之一：

1. `KCAR` 作为风险诊断器也不足以稳定预测偏离 RA 的机会窗口；
2. 当前 MI setting 下，真正可用的强默认策略仍然只有 `RA`；
3. 阶段三主结果应改写为：
   - `KCAR` 的局限性分析
   - 为什么 `RA` 仍是最稳默认策略
   - 哪类动态信号可能才值得继续研究

---

## 7. 与第二阶段策略实验的关系

`kcar_safe_policy.py` 只消费 `window_metrics.csv`：

- 不重新训练 backbone
- 只做 leave-one-subject-out threshold selection
- 只在 `w=1.0` 与 `w=0.5` 间选择

这样能保证第二阶段的策略结果完全建立在第一阶段的风险诊断之上，而不是重新引入新的模型变量。
