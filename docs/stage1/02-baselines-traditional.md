# 02 - 基线方法：传统方法

本文档包含三个传统的基线方法：CSP+LDA、Euclidean Alignment、Riemannian Alignment

---

## 🔧 Baseline 1: CSP + LDA

### 方法描述
- **CSP (Common Spatial Patterns)**：经典的MI特征提取方法
- **LDA (Linear Discriminant Analysis)**：线性分类器

### 理论基础

#### CSP原理
```
目标：找到空间滤波器，使得两类信号的方差比最大化

数学形式：
给定两类协方差矩阵 C1, C2
求解广义特征值问题：
C1 * W = λ * C2 * W

输出：
- 空间滤波器 W ∈ R^{n_channels × n_filters}
- 通常取前3个和后3个特征向量（共6个）

特征提取：
X_filtered = W^T @ X  # (n_filters, n_samples)
features = log(var(X_filtered, axis=1))  # (n_filters,)
```

**为什么CSP有效？**
- 左手MI：右侧运动皮层（C4）mu节律抑制
- 右手MI：左侧运动皮层（C3）mu节律抑制
- CSP自动找到最能区分这种模式的空间滤波器

### 实现细节

#### Step 1: 计算协方差矩阵
```python
# 对每个trial：
# X_trial: (n_channels, n_samples) = (22, 750)

# 协方差矩阵：
C = (X @ X.T) / np.trace(X @ X.T)  # 归一化

# 输出：
# C: (n_channels, n_channels) = (22, 22)
```

#### Step 2: 训练CSP
```python
# 输入：
# - X_train: (n_trials, n_channels, n_samples)
# - y_train: (n_trials,)

# 对于4类问题，使用 One-vs-Rest 策略：
# - CSP_0: 类别0 vs. 其他
# - CSP_1: 类别1 vs. 其他
# - CSP_2: 类别2 vs. 其他
# - CSP_3: 类别3 vs. 其他

# 每个CSP提取6个特征 → 总共24个特征
```

#### Step 3: 训练LDA分类器
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(
    solver='lsqr',      # 对小样本更稳定
    shrinkage='auto'    # 正则化，防止过拟合
)

lda.fit(features_train, y_train)
```

#### Step 4: 测试
```python
# 对测试集的每个trial：
# 1. 计算协方差矩阵
# 2. 用训练好的CSP提取特征
# 3. 用LDA预测类别

y_pred = lda.predict(features_test)
```

### 预期结果

**Within-subject（被试内）**：
- 训练集准确率：75-85%
- 测试集准确率：70-80%
- Kappa：0.6-0.75

**Cross-subject（跨被试，LOSO）**：
- 平均准确率：**40-50%**（接近随机，25%）
- 这就是为什么需要迁移学习！

**关键观察**：
- 某些被试对（如A01→A02）效果还可以（60%）
- 某些被试对（如A01→A05）效果很差（30%）
- 这就是"表征-行为不一致"的初步证据

### 代码模板
```python
# src/features/csp.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=6, reg=0.1):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None

    def fit(self, X, y):
        """
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,)
        """
        # 计算每类的平均协方差矩阵
        # 求解广义特征值问题
        # 保存滤波器
        pass

    def transform(self, X):
        """
        X: (n_trials, n_channels, n_samples)
        返回: (n_trials, n_components)
        """
        # 应用滤波器
        # 计算对数方差特征
        pass
```

---

## 🔧 Baseline 2: Euclidean Alignment + CSP + LDA

### 方法描述
- 在CSP之前，先用Euclidean Alignment对齐协方差矩阵
- 然后正常提取CSP特征和分类

### 理论基础

#### Euclidean Alignment (EA) 原理
```
目标：对齐源域和目标域的协方差矩阵的均值和方差

假设：
- 源域协方差矩阵：{C_s^1, ..., C_s^n}
- 目标域协方差矩阵：{C_t^1, ..., C_t^m}

Step 1: 计算均值
C_s_mean = (1/n) * Σ C_s^i
C_t_mean = (1/m) * Σ C_t^i

Step 2: 白化（Whitening）
对齐公式：
C_t_aligned^i = C_s_mean^{1/2} @ C_t_mean^{-1/2} @ C_t^i @ C_t_mean^{-1/2} @ C_s_mean^{1/2}

直觉：
- C_t_mean^{-1/2}：将目标域"白化"到单位矩阵
- C_s_mean^{1/2}：将白化后的数据"染色"成源域的分布
```

**为什么EA有效？**
- 不同被试的EEG信号幅度、信噪比不同
- EA通过对齐二阶统计量（协方差），减少这种差异
- 但EA假设线性关系，忽略了流形几何

### 实现细节

#### Step 1: 计算参考协方差矩阵
```python
# 源域（训练集）：
C_s_mean = np.mean(C_s_all, axis=0)  # 算术平均

# 目标域（测试集）：
# 方法A：用所有测试数据（离线）
C_t_mean = np.mean(C_t_all, axis=0)

# 方法B：用前K个trial估计（在线）
C_t_mean = np.mean(C_t_all[:K], axis=0)  # K=10-20
```

#### Step 2: 矩阵平方根和逆平方根
```python
def matrix_sqrt(C, inverse=False):
    """
    计算 C^{1/2} 或 C^{-1/2}
    """
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(C)

    # 正则化（防止数值问题）
    eigvals = np.maximum(eigvals, 1e-6)

    if inverse:
        eigvals = 1.0 / np.sqrt(eigvals)
    else:
        eigvals = np.sqrt(eigvals)

    # 重构
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
```

#### Step 3: 对齐每个trial
```python
def euclidean_alignment(C_t, C_s_mean, C_t_mean):
    """
    对齐单个协方差矩阵
    """
    C_t_mean_inv_sqrt = matrix_sqrt(C_t_mean, inverse=True)
    C_s_mean_sqrt = matrix_sqrt(C_s_mean, inverse=False)

    C_t_aligned = C_s_mean_sqrt @ C_t_mean_inv_sqrt @ C_t @ C_t_mean_inv_sqrt @ C_s_mean_sqrt

    return C_t_aligned
```

#### Step 4: 后续流程同Baseline 1
```python
# 对齐后的协方差矩阵 → CSP → LDA
```

### 预期结果

**Cross-subject（LOSO）**：
- 平均准确率：**55-65%**（比无对齐提升10-15%）
- 某些被试提升明显（如A01→A05：30%→55%）
- 某些被试提升不大（如A01→A02：60%→62%）

**关键观察**：
- EA对"幅度差异大"的被试对效果好
- 对"模式差异大"的被试对效果有限
- 这说明需要更复杂的对齐方法

### 代码模板
```python
# src/alignment/euclidean.py
class EuclideanAlignment:
    def __init__(self):
        self.C_ref = None  # 参考协方差矩阵

    def fit(self, C_source):
        """
        C_source: (n_trials, n_channels, n_channels)
        """
        self.C_ref = np.mean(C_source, axis=0)

    def transform(self, C_target):
        """
        C_target: (n_trials, n_channels, n_channels)
        返回: (n_trials, n_channels, n_channels)
        """
        C_target_mean = np.mean(C_target, axis=0)

        # 对齐每个trial
        C_aligned = []
        for C_t in C_target:
            C_t_aligned = self._align_single(C_t, self.C_ref, C_target_mean)
            C_aligned.append(C_t_aligned)

        return np.array(C_aligned)

    def _align_single(self, C_t, C_s_mean, C_t_mean):
        # 实现对齐公式
        pass
```

---

## 🔧 Baseline 3: Riemannian Alignment + CSP + LDA

### 方法描述
- 在黎曼流形上对齐协方差矩阵
- 比EA更尊重协方差矩阵的几何结构

### 理论基础

#### Riemannian Geometry 基础
```
协方差矩阵是对称正定（SPD）矩阵：
SPD(n) = {C ∈ R^{n×n} | C = C^T, C > 0}

SPD流形的性质：
- 不是欧几里得空间（不能直接相加、平均）
- 是黎曼流形（有内在的几何结构）
- 距离度量：Riemannian距离

Riemannian距离：
d_R(C1, C2) = ||Log(C1^{-1/2} @ C2 @ C1^{-1/2})||_F

其中 Log 是矩阵对数
```

#### Riemannian Mean（黎曼均值）
```
欧几里得均值（错误）：
C_mean = (1/n) * Σ C_i  # 可能不是正定的！

黎曼均值（正确）：
C_mean = argmin_C Σ d_R(C, C_i)^2

计算方法：迭代算法
1. 初始化：C_mean = (1/n) * Σ C_i
2. 重复直到收敛：
   a. 计算梯度：G = Σ Log(C_mean^{-1/2} @ C_i @ C_mean^{-1/2})
   b. 更新：C_mean = C_mean^{1/2} @ Exp(α*G) @ C_mean^{1/2}

其中 Exp 是矩阵指数，α 是步长
```

#### Riemannian Alignment (RA) 原理
```
类似EA，但在流形上操作：

Step 1: 计算黎曼均值
C_s_mean = RiemannianMean({C_s^1, ..., C_s^n})
C_t_mean = RiemannianMean({C_t^1, ..., C_t^m})

Step 2: 平行传输（Parallel Transport）
对齐公式：
C_t_aligned^i = ParallelTransport(C_t^i, from=C_t_mean, to=C_s_mean)

具体实现（简化版）：
C_t_aligned^i = C_s_mean^{1/2} @ C_t_mean^{-1/2} @ C_t^i @ C_t_mean^{-1/2} @ C_s_mean^{1/2}

注意：这个公式和EA一样，但C_mean的计算方式不同！
```

**为什么RA比EA好？**
- EA用算术平均，可能产生"不自然"的协方差矩阵
- RA用黎曼均值，保证结果仍在SPD流形上
- RA更好地保留了协方差矩阵的几何结构

### 实现细节

#### Step 1: 计算黎曼均值
```python
# 使用 PyRiemann 库：
from pyriemann.utils.mean import mean_riemann

C_s_mean = mean_riemann(C_s_all)  # C_s_all: (n_trials, n_channels, n_channels)
C_t_mean = mean_riemann(C_t_all)
```

#### Step 2: 对齐（同EA）
```python
# 对齐公式和EA相同，但C_mean不同
C_t_aligned = euclidean_alignment(C_t, C_s_mean, C_t_mean)
```

#### Step 3: 后续流程同Baseline 1

### 预期结果

**Cross-subject（LOSO）**：
- 平均准确率：**60-70%**（比EA再提升5%）
- 在BCI Competition IV 2a上，RA通常是最好的静态对齐方法

**关键观察**：
- RA在所有被试对上都比EA好或持平
- 但仍然有10-15%的gap到within-subject性能
- 这就是你的方法的改进空间

### 代码模板
```python
# src/alignment/riemannian.py
from pyriemann.utils.mean import mean_riemann

class RiemannianAlignment:
    def __init__(self):
        self.C_ref = None

    def fit(self, C_source):
        """
        C_source: (n_trials, n_channels, n_channels)
        """
        # 使用黎曼均值
        self.C_ref = mean_riemann(C_source)

    def transform(self, C_target):
        """
        C_target: (n_trials, n_channels, n_channels)
        """
        # 计算目标域的黎曼均值
        C_target_mean = mean_riemann(C_target)

        # 对齐（公式和EA相同）
        C_aligned = []
        for C_t in C_target:
            C_t_aligned = self._align_single(C_t, self.C_ref, C_target_mean)
            C_aligned.append(C_t_aligned)

        return np.array(C_aligned)

    def _align_single(self, C_t, C_s_mean, C_t_mean):
        # 和EA的对齐公式相同
        pass
```

---

## 📊 性能对比总结

| 方法 | Within-Subject | Cross-Subject (LOSO) | 提升 |
|------|----------------|---------------------|------|
| CSP+LDA | 70-80% | 40-50% | - |
| EA+CSP+LDA | - | 55-65% | +10-15% |
| RA+CSP+LDA | - | 60-70% | +15-20% |

---

## 🎯 实现优先级

1. **必须实现**: CSP+LDA（最基础）
2. **强烈推荐**: RA+CSP+LDA（最好的静态方法）
3. **可选**: EA+CSP+LDA（如果时间充裕）

---

## 📚 参考文献

- CSP原始论文：Ramoser et al. (2000) "Optimal spatial filtering of single trial EEG during imagined hand movement"
- EA论文：He & Wu (2019) "Transfer Learning for Brain-Computer Interfaces"
- RA论文：Zanini et al. (2018) "Transfer Learning: A Riemannian Geometry Framework With Applications to Brain-Computer Interfaces"
- PyRiemann文档：https://pyriemann.readthedocs.io/
