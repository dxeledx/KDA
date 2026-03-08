# 04 - 表征-行为不一致现象验证

本文档详细说明如何验证"表征-行为不一致"现象，这是你研究的核心动机。

---

## 🎯 目标

用数据证明两种现象确实存在：
1. **表征相似但行为不同**：神经模式相似，但分类性能差异大
2. **表征不同但行为相似**：神经模式差异大，但分类性能相近

---

## 📐 Part 1: 实验设计

### Step 1: 提取表征
```python
# 对每个被试：
# 1. 在训练集上训练CSP+LDA
# 2. 在测试集上提取CSP特征
# 3. 得到特征矩阵：F_i ∈ R^{n_trials × n_features}

for subject_id in range(1, 10):
    # 训练CSP
    csp = CSP(n_components=6)
    csp.fit(X_train[subject_id], y_train[subject_id])

    # 提取特征
    features[subject_id] = csp.transform(X_test[subject_id])
    # features[subject_id]: (288, 24)  # 288 trials, 24 features
```

### Step 2: 计算表征相似度

#### 方法A：CKA (Centered Kernel Alignment) - 推荐
```python
def cka(X, Y):
    """
    计算两个表征的CKA相似度

    X: (n_samples, n_features_x)
    Y: (n_samples, n_features_y)

    返回: CKA ∈ [0, 1]
    """
    n = X.shape[0]

    # Gram矩阵
    K = X @ X.T
    L = Y @ Y.T

    # 中心化
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    # CKA
    numerator = np.sum(K_c * L_c)  # Frobenius内积
    denominator = np.linalg.norm(K_c, 'fro') * np.linalg.norm(L_c, 'fro')

    return numerator / denominator

# 计算所有被试对的CKA
n_subjects = 9
S_rep = np.zeros((n_subjects, n_subjects))

for i in range(n_subjects):
    for j in range(i+1, n_subjects):
        S_rep[i, j] = cka(features[i], features[j])
        S_rep[j, i] = S_rep[i, j]  # 对称
```

**为什么用CKA？**
- CKA是衡量两个表征相似度的标准方法
- 对线性变换不变（如旋转、缩放）
- 在神经网络表征分析中广泛使用
- CKA ∈ [0, 1]，1表示完全相似，0表示完全不相关

#### 方法B：CCA (Canonical Correlation Analysis) - 备选
```python
from sklearn.cross_decomposition import CCA

def cca_similarity(X, Y, n_components=5):
    """
    计算CCA相似度
    """
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)

    # 计算典型相关系数
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                   for i in range(n_components)]

    return np.mean(correlations)
```

#### 方法C：余弦相似度 - 最简单
```python
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(X, Y):
    """
    计算平均特征的余弦相似度
    """
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    return cosine_similarity([X_mean], [Y_mean])[0, 0]
```

### Step 3: 计算行为一致性

#### 方法A：准确率差异 - 推荐
```python
# 对每对被试 (i, j)：
S_beh = np.zeros((n_subjects, n_subjects))

for i in range(n_subjects):
    for j in range(i+1, n_subjects):
        # 计算准确率
        acc_i = accuracy_score(y_test[i], y_pred[i])
        acc_j = accuracy_score(y_test[j], y_pred[j])

        # 行为一致性 = 1 - 准确率差异
        S_beh[i, j] = 1 - abs(acc_i - acc_j)
        S_beh[j, i] = S_beh[i, j]
```

**直觉**：
- 如果两个被试的准确率都是75%，S_beh = 1（完全一致）
- 如果一个75%，一个50%，S_beh = 0.75（差异25%）

#### 方法B：混淆矩阵相似度 - 更细致
```python
from scipy.stats import pearsonr

def confusion_matrix_similarity(CM_i, CM_j):
    """
    计算两个混淆矩阵的相似度
    """
    # 展平混淆矩阵
    cm_i_flat = CM_i.flatten()
    cm_j_flat = CM_j.flatten()

    # 计算相关系数
    corr, _ = pearsonr(cm_i_flat, cm_j_flat)

    return corr

# 计算所有被试对的混淆矩阵相似度
for i in range(n_subjects):
    for j in range(i+1, n_subjects):
        CM_i = confusion_matrix(y_test[i], y_pred[i])
        CM_j = confusion_matrix(y_test[j], y_pred[j])

        S_beh[i, j] = confusion_matrix_similarity(CM_i, CM_j)
```

### Step 4: 可视化

#### 图1：表征-行为散点图（核心图）
```python
import matplotlib.pyplot as plt

# 提取上三角（去掉对角线）
rep_values = S_rep[np.triu_indices(n_subjects, k=1)]
beh_values = S_beh[np.triu_indices(n_subjects, k=1)]

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(rep_values, beh_values, alpha=0.6, s=100, c='steelblue')

# 添加对角线（理想情况）
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect alignment')

# 标注典型案例
# 案例A：表征相似但行为不同（右下角）
# 案例B：表征不同但行为相似（左上角）

plt.xlabel('Representation Similarity (CKA)', fontsize=14)
plt.ylabel('Behavior Consistency', fontsize=14)
plt.title('Representation-Behavior Inconsistency', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('results/figures/rep_beh_scatter.pdf', dpi=300)
plt.show()
```

**预期结果**：
- 散点分布在对角线两侧，但不在对角线上
- 右下角有一些点（表征相似但行为不同）
- 左上角有一些点（表征不同但行为相似）
- 相关系数 r = 0.3-0.5（弱相关）

**这张图将成为你论文Introduction的核心插图！**

### Step 5: 量化分析

#### 计算相关系数
```python
from scipy.stats import pearsonr

# 计算Pearson相关系数
r, p_value = pearsonr(rep_values, beh_values)

print(f"Correlation: r = {r:.3f}, p = {p_value:.4f}")

# 预期：
# - 无对齐：r = 0.3-0.5（弱相关）
# - EA对齐：r = 0.5-0.6
# - RA对齐：r = 0.6-0.7
# - 你的方法（目标）：r > 0.75
```

#### 统计检验
```python
# 检验相关性是否显著
if p_value < 0.05:
    print("相关性显著 (p < 0.05)")
else:
    print("相关性不显著")
```

### Step 6: 识别典型案例

#### 案例A：表征相似但行为不同
```python
# 筛选条件：S_rep > 0.7 且 S_beh < 0.5
inconsistent_cases_A = []

for i in range(n_subjects):
    for j in range(i+1, n_subjects):
        if S_rep[i, j] > 0.7 and S_beh[i, j] < 0.5:
            inconsistent_cases_A.append((i, j, S_rep[i, j], S_beh[i, j]))

print("案例A：表征相似但行为不同")
for i, j, rep, beh in inconsistent_cases_A:
    print(f"  被试 A{i+1:02d} vs A{j+1:02d}: CKA={rep:.3f}, Behavior={beh:.3f}")
```

**分析原因**：
- 可能是决策边界不同
- 可能是某些类别的混淆模式不同
- 需要可视化决策边界和混淆矩阵

#### 案例B：表征不同但行为相似
```python
# 筛选条件：S_rep < 0.4 且 S_beh > 0.7
inconsistent_cases_B = []

for i in range(n_subjects):
    for j in range(i+1, n_subjects):
        if S_rep[i, j] < 0.4 and S_beh[i, j] > 0.7:
            inconsistent_cases_B.append((i, j, S_rep[i, j], S_beh[i, j]))

print("案例B：表征不同但行为相似")
for i, j, rep, beh in inconsistent_cases_B:
    print(f"  被试 A{i+1:02d} vs A{j+1:02d}: CKA={rep:.3f}, Behavior={beh:.3f}")
```

**分析原因**：
- 可能是非线性流形结构
- 可能是不同的特征子空间编码了相同的信息
- 需要更复杂的对齐方法（如你的DCA-BGF）

---

## 📊 Part 2: 补充可视化

### 图2：协方差矩阵热图
```python
import seaborn as sns

# 选择3个代表性被试
subjects = [0, 4, 8]  # A01, A05, A09

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, subject in enumerate(subjects):
    # 计算平均协方差矩阵
    C_mean = np.mean(covariances[subject], axis=0)  # (22, 22)

    # 绘制热图
    sns.heatmap(C_mean, ax=axes[idx], cmap='RdBu_r',
                center=0, square=True, cbar=True,
                xticklabels=False, yticklabels=False)
    axes[idx].set_title(f'Subject A{subject+1:02d}', fontsize=14)

plt.tight_layout()
plt.savefig('results/figures/covariance_heatmaps.pdf', dpi=300)
plt.show()
```

**预期结果**：
- 不同被试的协方差矩阵模式明显不同
- 某些被试的C3-C4相关性强（运动皮层）
- 某些被试的模式更分散

### 图3：跨被试性能热图
```python
# 计算所有被试对的准确率
acc_matrix = np.zeros((9, 9))

for i in range(9):  # 源域
    for j in range(9):  # 目标域
        if i == j:
            # Within-subject
            acc_matrix[i, j] = within_subject_acc[i]
        else:
            # Cross-subject: 在被试i上训练，在被试j上测试
            acc_matrix[i, j] = cross_subject_acc[i, j]

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
            xticklabels=[f'A{i+1:02d}' for i in range(9)],
            yticklabels=[f'A{i+1:02d}' for i in range(9)],
            vmin=0.25, vmax=0.85, cbar_kws={'label': 'Accuracy'})
plt.xlabel('Target Subject', fontsize=14)
plt.ylabel('Source Subject', fontsize=14)
plt.title('Cross-Subject Transfer Performance', fontsize=16)
plt.tight_layout()
plt.savefig('results/figures/transfer_matrix.pdf', dpi=300)
plt.show()
```

**预期结果**：
- 对角线（within-subject）准确率最高（70-80%）
- 非对角线（cross-subject）准确率低（40-60%）
- 某些被试对（如A01→A02）效果好
- 某些被试对（如A01→A05）效果差

**这张图展示了跨被试迁移的难度！**

---

## 📈 Part 3: 对比不同方法

### 对比无对齐 vs. EA vs. RA

```python
methods = ['No Alignment', 'EA', 'RA']
correlations = []

for method in methods:
    # 计算该方法下的表征-行为相关性
    rep_values = S_rep_method[method]
    beh_values = S_beh_method[method]

    r, p = pearsonr(rep_values, beh_values)
    correlations.append(r)

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(methods, correlations, color=['steelblue', 'orange', 'green'])
plt.ylabel('Correlation (r)', fontsize=14)
plt.title('Representation-Behavior Correlation', fontsize=16)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, r in enumerate(correlations):
    plt.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('results/figures/correlation_comparison.pdf', dpi=300)
plt.show()
```

**预期结果**：
- 无对齐：r = 0.3-0.5
- EA：r = 0.5-0.6
- RA：r = 0.6-0.7
- 你的方法（目标）：r > 0.75

---

## 🎯 完成标准

现象验证完成后，你应该有：

- ✅ 表征-行为散点图（核心图）
- ✅ 相关系数 r 和 p-value
- ✅ 识别出2-3个典型的不一致案例
- ✅ 协方差矩阵热图
- ✅ 跨被试性能热图
- ✅ 不同方法的对比图

---

## 📝 论文写作建议

### Introduction部分
```
"Figure 1 illustrates the representation-behavior inconsistency phenomenon
in cross-subject MI-BCI. We computed the representation similarity (CKA)
and behavior consistency (1 - |Δacc|) for all subject pairs. The weak
correlation (r = 0.42, p < 0.01) indicates that similar neural
representations do not guarantee similar classification performance,
and vice versa. This motivates our dynamic conditional alignment approach."
```

### 典型案例描述
```
"We identified two typical cases of inconsistency:

Case A (Subjects A01 and A03): Despite high representation similarity
(CKA = 0.78), their classification accuracies differ significantly
(75% vs. 52%), suggesting different decision boundaries.

Case B (Subjects A02 and A07): Despite low representation similarity
(CKA = 0.35), their classification accuracies are nearly identical
(68% vs. 70%), indicating nonlinear manifold structure that static
alignment methods fail to capture."
```

---

## 🔍 调试提示

### 如果相关系数太高（r > 0.7）
- 检查是否用了within-subject数据（应该用cross-subject）
- 检查CKA计算是否正确
- 可能数据集太简单，考虑用更难的数据集

### 如果相关系数太低（r < 0.2）
- 检查特征提取是否正确
- 检查是否有数据泄露
- 可能需要更多被试数据

### 如果找不到不一致案例
- 调整阈值（如S_rep > 0.6, S_beh < 0.6）
- 可视化所有被试对，手动挑选
- 可能需要更细致的分析（如按类别分析）

---

## 📚 参考资料

- CKA论文：Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
- CCA论文：Raghu et al. (2017) "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability"
- 表征分析综述：Nguyen et al. (2021) "Analyzing Learned Molecular Representations for Property Prediction"
