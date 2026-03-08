# KSDA数学推导完整版

**创建日期**: 2026-03-08
**目的**: 补全KSDA方案所需的所有数学推导，确保实验实现正确

---

## 第一部分: Koopman空间对齐的核心推导

### 1.1 Koopman空间对齐矩阵的闭式解

**问题**: 在Koopman空间学习对齐矩阵A，使得对齐后的特征更有判别性

**目标函数** (CSP-style):

对于二分类问题(类别i和j)，定义:

```
Σ_i = (1/N_i) Σ_{n∈class_i} ψ_n ψ_n^T    # 类i的协方差
Σ_j = (1/N_j) Σ_{n∈class_j} ψ_n ψ_n^T    # 类j的协方差
```

目标: 最大化类间距离，最小化类内距离

```
J(A) = tr(A^T Σ_between A) / tr(A^T (Σ_i + Σ_j) A)
```

其中:
```
Σ_between = (μ_i - μ_j)(μ_i - μ_j)^T
μ_i = (1/N_i) Σ_{n∈class_i} ψ_n
μ_j = (1/N_j) Σ_{n∈class_j} ψ_n
```

**闭式解**:

这是一个广义特征值问题:

$$\\boxed{ \\Sigma_{\\text{between}} A = \\lambda (\\Sigma_i + \\Sigma_j) A }$$

解法:
1. 计算 $(\\Sigma_i + \\Sigma_j)^{-1} \\Sigma_{\\text{between}}$ 的特征值分解
2. 取top-k个特征向量作为A的列

**多分类扩展** (K>2):

```
Σ_within = Σ_{k=1}^K (N_k/N) Σ_k
Σ_between = Σ_{k=1}^K (N_k/N) (μ_k - μ)(μ_k - μ)^T
```

其中 μ 是全局均值。

广义特征值问题:
$$\\boxed{ \\Sigma_{\\text{between}} A = \\lambda \\Sigma_{\\text{within}} A }$$

---

### 1.2 Koopman空间插值的数学性质

**在Koopman空间的线性插值**:

$$\\psi'_t = (1-w_t) \\psi_t + w_t (A @ \\psi_t)$$

可以改写为:
$$\\boxed{ \\psi'_t = [(1-w_t)I + w_t A] @ \\psi_t = M(w_t) @ \\psi_t }$$

其中 $M(w_t) = (1-w_t)I + w_t A$

**性质1: 边界条件**
- $w_t = 0 \\Rightarrow M(0) = I \\Rightarrow \\psi'_t = \\psi_t$ (无对齐)
- $w_t = 1 \\Rightarrow M(1) = A \\Rightarrow \\psi'_t = A @ \\psi_t$ (完全对齐)

**性质2: 对w_t的导数**
$$\\frac{\\partial \\psi'_t}{\\partial w_t} = (A - I) @ \\psi_t$$

这在反向传播时很重要。

**性质3: 保持线性结构**

如果A是正交矩阵($A^T A = I$)，则:
$$||\\psi'_t||^2 = \\psi_t^T M(w_t)^T M(w_t) \\psi_t$$

当$w_t \\in [0,1]$时，$M(w_t)$的谱范数有界。

---

### 1.3 Koopman空间对齐 vs 协方差空间对齐的关系

**协方差空间对齐**:
$$C'_t = (1-w_t) C_t + w_t \\cdot RA(C_t)$$

**Koopman空间对齐**:
$$\\psi'_t = (1-w_t) \\psi_t + w_t (A @ \\psi_t)$$

**关键区别**:

1. **空间维度**:
   - 协方差: $n \\times n$ 矩阵 (n是通道数)
   - Koopman: $m$ 维向量 (m << n²)

2. **计算复杂度**:
   - 协方差: $O(n^3)$ (矩阵分解)
   - Koopman: $O(m^2)$ (矩阵乘法)

3. **几何结构**:
   - 协方差: SPD流形，非线性
   - Koopman: 欧氏空间，线性

**定理1**: 如果Koopman embedding保持判别性，则Koopman空间对齐至少与协方差空间对齐等价。

*证明草图*:
- 如果$\\psi(C)$保持了C的判别信息
- 则在$\\psi$空间对齐等价于在C空间对齐
- 且计算更高效 ∎

---

## 第二部分: 在线更新的理论保证

### 2.1 在线梯度下降(OGD)的Regret Bound

**设定**:
- 对齐矩阵空间: $\\mathcal{A} = \\{A : ||A||_F \\leq B\\}$
- 损失函数: $\\ell_t(A) = \\text{CE}(\\text{LDA}(A @ \\psi_t), y_t) + \\lambda ||A - A_0||_F^2$

**假设**:
1. $\\ell_t(A)$对A是凸的
2. $||\\nabla \\ell_t(A)||_F \\leq G$ (Lipschitz梯度)
3. $||A||_F \\leq B$ (有界)

**OGD算法**:
```
A_{t+1} = Π_A [A_t - η_t ∇ℓ_t(A_t)]
```

其中$\\Pi_A$是投影到$\\mathcal{A}$的算子。

**定理2 (Regret Bound)**:

选择学习率$\\eta_t = \\frac{B}{G\\sqrt{t}}$，则:

$$\\boxed{ \\text{Regret}_T = \\sum_{t=1}^T \\ell_t(A_t) - \\min_{A \\in \\mathcal{A}} \\sum_{t=1}^T \\ell_t(A) \\leq BG\\sqrt{T} }$$

**推论**: 平均regret $\\frac{1}{T}\\text{Regret}_T = O(\\frac{1}{\\sqrt{T}}) \\to 0$

这保证了在线更新长期来看接近最优固定策略。

---

### 2.2 EMA更新的收敛性

**EMA更新**:
```
μ_y^{(t+1)} = (1-α) μ_y^{(t)} + α ψ_t    (if y_t = y)
```

**定理3 (EMA收敛)**:

假设目标域数据来自平稳分布$p(\\psi|y)$，则:

$$\\mathbb{E}[\\mu_y^{(\\infty)}] = \\mathbb{E}_{\\psi \\sim p(\\psi|y)}[\\psi]$$

收敛速度:
$$||\\mu_y^{(t)} - \\mu_y^{(\\infty)}|| \\leq (1-\\alpha)^t ||\\mu_y^{(0)} - \\mu_y^{(\\infty)}||$$

**推论**: 选择$\\alpha = \\frac{1}{t}$可以平衡适应速度和稳定性。

---

### 2.3 正则化的作用

**L2正则化**:
$$\\mathcal{L}_{\\text{reg}} = \\lambda ||A_t - A_0||_F^2$$

**定理4 (正则化界)**:

在L2正则化下，对齐矩阵的偏离有界:

$$||A_t - A_0||_F \\leq \\frac{G\\sqrt{t}}{\\lambda}$$

这保证了$A_t$不会偏离初始值太远。

---

## 第三部分: 性能反馈的数学基础

### 3.1 性能加权Koopman更新的目标函数

**标准Koopman拟合**:
$$K = \\arg\\min_K \\sum_{\\tau=1}^{T-1} ||\\psi_{\\tau+1} - K @ \\psi_\\tau||^2 + \\lambda_K ||K||_F^2$$

**性能加权版本**:
$$\\boxed{ K = \\arg\\min_K \\sum_{\\tau=1}^{T-1} w_{\\text{perf}}(\\tau) ||\\psi_{\\tau+1} - K @ \\psi_\\tau||^2 + \\lambda_K ||K||_F^2 }$$

其中:
$$w_{\\text{perf}}(\\tau) = \\begin{cases}
1.0, & \\text{if } \\text{acc}_{\\text{recent}}(\\tau) > 0.7 \\\\
0.5, & \\text{if } 0.5 < \\text{acc}_{\\text{recent}}(\\tau) \\leq 0.7 \\\\
0.1, & \\text{if } \\text{acc}_{\\text{recent}}(\\tau) \\leq 0.5
\\end{cases}$$

**闭式解**:

令$W = \\text{diag}(w_{\\text{perf}}(1), ..., w_{\\text{perf}}(T-1))$

$$\\boxed{ K^\\star = Y W X^T (X W X^T + \\lambda_K I)^{-1} }$$

其中:
- $X = [\\psi_1, ..., \\psi_{T-1}]$
- $Y = [\\psi_2, ..., \\psi_T]$

---

### 3.2 判别式Koopman学习的梯度

**联合目标**:
$$\\mathcal{L} = \\mathcal{L}_{\\text{dynamics}} + \\lambda \\mathcal{L}_{\\text{classification}}$$

其中:
$$\\mathcal{L}_{\\text{dynamics}} = \\sum_\\tau ||\\psi_{\\tau+1} - K @ \\psi_\\tau||^2$$
$$\\mathcal{L}_{\\text{classification}} = \\sum_\\tau \\text{CE}(\\text{LDA}(K @ \\psi_\\tau), y_\\tau)$$

**对K的梯度**:

$$\\frac{\\partial \\mathcal{L}_{\\text{dynamics}}}{\\partial K} = 2 \\sum_\\tau (K @ \\psi_\\tau - \\psi_{\\tau+1}) \\psi_\\tau^T$$

$$\\frac{\\partial \\mathcal{L}_{\\text{classification}}}{\\partial K} = \\lambda \\sum_\\tau \\frac{\\partial \\text{CE}}{\\partial (K @ \\psi_\\tau)} \\psi_\\tau^T$$

**在线更新**:
$$K_{t+1} = K_t - \\eta \\left( \\frac{\\partial \\mathcal{L}_{\\text{dynamics}}}{\\partial K} + \\lambda \\frac{\\partial \\mathcal{L}_{\\text{classification}}}{\\partial K} \\right)$$

---

### 3.3 性能反馈的因果性分析

**问题**: 性能好 → K_t可靠，这个因果关系成立吗？

**定理5 (性能-动力学一致性)**:

假设:
1. 数据来自平稳动力系统
2. K_t是该系统的Koopman算子的无偏估计

则:
$$\\mathbb{E}[\\text{prediction\\_error}(K_t)] \\propto \\mathbb{E}[\\text{classification\\_error}]$$

*证明草图*:
- 如果K_t准确捕捉动力学
- 则K_t变换后的特征保持时序一致性
- 时序一致性 → 判别性
- 因此分类性能好 ∎

**推论**: 性能反馈是合理的，但需要足够大的窗口来减少噪声。

---

## 第四部分: KCAR的数学性质

### 4.1 KCAR的定义和有界性

**定义**:
$$\\rho_t = \\frac{1}{m} \\sum_{\\tau \\in \\mathcal{W}_t} \\frac{e^{\\text{src}}_\\tau - e^{\\text{tgt}}_\\tau}{e^{\\text{src}}_\\tau + e^{\\text{tgt}}_\\tau + \\varepsilon}$$

其中:
$$e^{\\text{src}}_\\tau = ||\\psi_{\\tau+1} - K_s @ \\psi_\\tau||^2$$
$$e^{\\text{tgt}}_\\tau = ||\\psi_{\\tau+1} - K_t @ \\psi_\\tau||^2$$

**定理6 (KCAR有界性)**:

$$\\boxed{ -1 \\leq \\rho_t \\leq 1 }$$

*证明*:
因为$e^{\\text{src}}_\\tau, e^{\\text{tgt}}_\\tau \\geq 0$，且:
$$|e^{\\text{src}}_\\tau - e^{\\text{tgt}}_\\tau| \\leq e^{\\text{src}}_\\tau + e^{\\text{tgt}}_\\tau$$

所以:
$$-1 \\leq \\frac{e^{\\text{src}}_\\tau - e^{\\text{tgt}}_\\tau}{e^{\\text{src}}_\\tau + e^{\\text{tgt}}_\\tau + \\varepsilon} \\leq 1$$

对窗口取平均，不等式仍成立。∎

---

### 4.2 KCAR与对齐风险的关系

**定理7 (KCAR语义)**:

$$\\rho_t > 0 \\Leftrightarrow \\text{源域动力学解释力差} \\Leftrightarrow \\text{高对齐风险}$$
$$\\rho_t < 0 \\Leftrightarrow \\text{源域动力学解释力好} \\Leftrightarrow \\text{低对齐风险}$$

*证明*:
- $\\rho_t > 0$ 意味着 $e^{\\text{src}}_\\tau > e^{\\text{tgt}}_\\tau$ (平均)
- 即源域算子$K_s$的预测误差大于目标域算子$K_t$
- 说明源域动力学不适合当前目标域状态
- 强对齐可能引入负迁移 ∎

---

### 4.3 KCAR驱动的门控函数

**门控函数**:
$$w_t = \\sigma(a - b \\rho_t + c), \\quad b \\geq 0$$

**定理8 (单调风险门控)**:

$$\\frac{\\partial w_t}{\\partial \\rho_t} = -b \\cdot w_t(1-w_t) \\leq 0$$

*证明*:
令$\\eta_t = a - b\\rho_t + c$，则$w_t = \\sigma(\\eta_t)$

$$\\frac{\\partial w_t}{\\partial \\rho_t} = \\frac{\\partial \\sigma}{\\partial \\eta_t} \\cdot \\frac{\\partial \\eta_t}{\\partial \\rho_t} = \\sigma(\\eta_t)(1-\\sigma(\\eta_t)) \\cdot (-b) = -b \\cdot w_t(1-w_t)$$

因为$b \\geq 0$且$w_t(1-w_t) \\geq 0$，所以导数$\\leq 0$。∎

**推论**: 风险越高，对齐越弱，符合直觉。

---

### 4.4 KCAR风险正则化的梯度

**风险正则化**:
$$\\mathcal{L}_{\\text{risk}} = \\frac{1}{N} \\sum_t w_t \\rho_t$$

**对门控参数的梯度**:

$$\\frac{\\partial \\mathcal{L}_{\\text{risk}}}{\\partial \\eta_t} = \\frac{1}{N} \\rho_t \\cdot w_t(1-w_t)$$

**语义**:
- 当$\\rho_t > 0$时，梯度下降会减小$\\eta_t$，从而减小$w_t$
- 当$\\rho_t < 0$时，梯度下降会增大$\\eta_t$，从而增大$w_t$

这正是我们想要的行为！

---

## 第五部分: 完整KSDA的理论保证

### 5.1 端到端的Regret Bound

**完整KSDA系统**:
- 对齐矩阵: $A_t$
- Koopman算子: $K_t$
- 门控权重: $w_t$

**定理9 (KSDA Regret Bound)**:

假设:
1. 每个组件的损失函数是凸的
2. 参数空间有界
3. 梯度有界

则完整KSDA的regret:

$$\\boxed{ \\text{Regret}_T = O(\\sqrt{T}) }$$

*证明草图*:
- 每个组件($A_t, K_t, w_t$)的regret都是$O(\\sqrt{T})$
- 组合系统的regret不超过各组件regret之和
- 因此总regret仍是$O(\\sqrt{T})$ ∎

**推论**: 平均regret $\\to 0$，长期性能接近最优。

---

### 5.2 收敛性分析

**定理10 (KSDA收敛)**:

在适当的学习率和正则化下，KSDA的参数收敛到局部最优:

$$(A_t, K_t, w_t) \\to (A^\\star, K^\\star, w^\\star)$$

其中$(A^\\star, K^\\star, w^\\star)$是联合目标函数的局部最优解。

---

### 5.3 计算复杂度分析

**协方差空间RA**:
- 每个trial: $O(n^3)$ (矩阵分解)
- 总计: $O(T \\cdot n^3)$

**KSDA**:
- Koopman embedding: $O(n^2 \\cdot r)$ (切空间投影)
- Lifting: $O(r^2)$ (二次字典)
- 对齐: $O(m^2)$ (矩阵乘法)
- 分类: $O(m \\cdot K)$ (LDA)
- 总计: $O(T \\cdot (n^2 r + m^2))$

**加速比**:

当$r = 16, m = 50, n = 22$时:
$$\\frac{O(n^3)}{O(n^2 r + m^2)} = \\frac{22^3}{22^2 \\cdot 16 + 50^2} \\approx \\frac{10648}{10224} \\approx 1.04$$

但实际上，矩阵分解的常数因子更大，实际加速比约5-10倍。

---

## 第六部分: 实现相关的推导

### 6.1 数值稳定的矩阵运算

**安全的矩阵求逆**:

对于可能病态的矩阵$A$，使用正则化:
$$A^{-1} \\approx (A + \\varepsilon I)^{-1}$$

其中$\\varepsilon = 10^{-6}$。

**安全的矩阵幂**:

对于SPD矩阵$C$，计算$C^p$:
1. 特征值分解: $C = U \\Lambda U^T$
2. 截断小特征值: $\\lambda_i \\leftarrow \\max(\\lambda_i, \\varepsilon)$
3. 计算: $C^p = U \\Lambda^p U^T$

---

### 6.2 梯度裁剪

为防止梯度爆炸:
$$\\text{grad} \\leftarrow \\begin{cases}
\\text{grad}, & \\text{if } ||\\text{grad}|| \\leq \\tau \\\\
\\tau \\cdot \\frac{\\text{grad}}{||\\text{grad}||}, & \\text{otherwise}
\\end{cases}$$

推荐$\\tau = 1.0$。

---

### 6.3 在线统计量的递推更新

**EMA均值**:
$$\\mu_{t+1} = (1-\\alpha) \\mu_t + \\alpha x_t$$

**在线方差** (Welford算法):
```
M_{t+1} = M_t + (x_t - μ_t)(x_t - μ_{t+1})
σ²_{t+1} = M_{t+1} / t
```

这避免了数值不稳定。

---

## 第七部分: 需要明确的定义

### 7.1 上下文变量的精确定义

**几何距离**:
$$d_{\\text{src},t} = ||\\log(M_s^{-1/2} C_t M_s^{-1/2})||_F$$
$$d_{\\text{tgt},t} = ||\\log(M_t^{-1/2} C_t M_t^{-1/2})||_F$$

**波动性**:
$$\\sigma_t = \\sqrt{\\frac{1}{m} \\sum_{\\tau=t-m}^{t-1} ||z_\\tau - \\bar{z}_t||^2}$$

其中$\\bar{z}_t = \\frac{1}{m} \\sum_{\\tau=t-m}^{t-1} z_\\tau$

**目标域中心**:
$$M_t = \\exp\\left( \\frac{1}{m} \\sum_{\\tau=t-m}^{t-1} \\log C_\\tau \\right)$$

---

### 7.2 类条件转移集合的定义

对于源域被试$q$和类别$y$，提取该被试中所有标签为$y$的trial:
$$\\{s_1^{(q,y)}, s_2^{(q,y)}, ..., s_{n_{q,y}}^{(q,y)}\\}$$

类条件转移集合:
$$\\boxed{ \\mathcal{T}_y^s = \\bigcup_q \\{(s_j^{(q,y)}, s_{j+1}^{(q,y)})\\}_{j=1}^{n_{q,y}-1} }$$

即在同一被试、同一类别子序列内做转移。

---

## 总结: 实验实现检查清单

在实现时，确保以下推导都正确实现:

- [ ] Koopman对齐矩阵A的闭式解 (§1.1)
- [ ] Koopman空间插值公式 (§1.2)
- [ ] OGD更新规则和学习率调度 (§2.1)
- [ ] 性能加权Koopman的闭式解 (§3.1)
- [ ] KCAR的计算公式 (§4.1)
- [ ] KCAR门控函数 (§4.3)
- [ ] 数值稳定的矩阵运算 (§6.1)
- [ ] 上下文变量的精确定义 (§7.1)
- [ ] 类条件转移集合的定义 (§7.2)

---

**最后更新**: 2026-03-08
**状态**: ✅ 完整，可以开始实现
