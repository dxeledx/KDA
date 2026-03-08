# 06 - 调试指南

本文档提供详细的调试清单和常见问题解决方案。

---

## ✅ Part 1: 数据验证清单

### 在开始训练之前，确保：

#### 1. 数据加载正确
```python
# 检查数据形状
print(f"X_train shape: {X_train.shape}")  # 应该是 (n_trials, n_channels, n_samples)
print(f"y_train shape: {y_train.shape}")  # 应该是 (n_trials,)

# 期望输出：
# X_train shape: (288, 22, 750)
# y_train shape: (288,)

# 检查标签范围
print(f"Labels: {np.unique(y_train)}")  # 应该是 [0, 1, 2, 3]

# 检查类别平衡
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
# 期望输出：{0: 72, 1: 72, 2: 72, 3: 72}
```

#### 2. 预处理正确
```python
# 检查滤波频段
print(f"Filter range: {l_freq}-{h_freq} Hz")  # 应该是 8-30 Hz

# 检查Epoch时间窗口
print(f"Epoch window: {tmin}-{tmax} s")  # 应该是 3-6 s
print(f"Samples per epoch: {n_samples}")  # 应该是 750 @ 250Hz

# 检查无NaN或Inf值
assert not np.isnan(X_train).any(), "Data contains NaN!"
assert not np.isinf(X_train).any(), "Data contains Inf!"
```

#### 3. 协方差矩阵正确
```python
# 计算协方差矩阵
C = compute_covariance(X_trial)

# 检查形状
assert C.shape == (22, 22), f"Wrong shape: {C.shape}"

# 检查对称性
assert np.allclose(C, C.T), "Covariance matrix not symmetric!"

# 检查正定性
eigvals = np.linalg.eigvalsh(C)
assert np.all(eigvals > 0), f"Not positive definite! Min eigval: {eigvals.min()}"

# 检查归一化
trace = np.trace(C)
print(f"Trace: {trace}")  # 应该接近1（如果归一化了）
```

#### 4. 数据分割正确
```python
# 检查训练/测试集大小
print(f"Train size: {len(X_train)}")  # 288
print(f"Test size: {len(X_test)}")    # 288

# 检查无数据泄露
# 训练和测试的trial应该完全不同
train_indices = set(range(len(X_train)))
test_indices = set(range(len(X_train), len(X_train) + len(X_test)))
assert train_indices.isdisjoint(test_indices), "Data leakage!"
```

---

## 🔧 Part 2: Baseline验证清单

### CSP+LDA验证

#### 1. Within-subject性能
```python
# 训练
csp = CSP(n_components=6)
lda = LDA(solver='lsqr', shrinkage='auto')

features_train = csp.fit_transform(X_train, y_train)
lda.fit(features_train, y_train)

# 测试
features_test = csp.transform(X_test)
y_pred = lda.predict(features_test)

# 评估
train_acc = lda.score(features_train, y_train)
test_acc = accuracy_score(y_test, y_pred)

print(f"Train accuracy: {train_acc:.3f}")  # 应该 75-85%
print(f"Test accuracy: {test_acc:.3f}")    # 应该 70-80%

# 如果低于60%，检查CSP实现
if test_acc < 0.6:
    print("WARNING: Accuracy too low! Check CSP implementation.")
```

#### 2. CSP滤波器可视化
```python
import matplotlib.pyplot as plt

# 获取CSP滤波器
filters = csp.filters_  # (n_channels, n_components)

# 画出前3个和后3个滤波器的空间模式
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i in range(3):
    # 前3个
    pattern = filters[:, i]
    axes[0, i].bar(range(22), pattern)
    axes[0, i].set_title(f'Filter {i+1}')

    # 后3个
    pattern = filters[:, -(i+1)]
    axes[1, i].bar(range(22), pattern)
    axes[1, i].set_title(f'Filter {-i-1}')

plt.tight_layout()
plt.savefig('results/figures/csp_filters.pdf')
plt.show()

# 应该看到C3/C4区域的激活
# 如果模式随机，说明CSP有问题
```

#### 3. 特征分布
```python
# 画出CSP特征的直方图
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i in range(6):
    ax = axes[i // 3, i % 3]

    for label in range(4):
        mask = y_train == label
        ax.hist(features_train[mask, i], alpha=0.5, label=f'Class {label}')

    ax.set_title(f'Feature {i+1}')
    ax.legend()

plt.tight_layout()
plt.savefig('results/figures/feature_distributions.pdf')
plt.show()

# 不同类别应该有分离
# 如果完全重叠，说明特征无效
```

### EA/RA验证

#### 1. 对齐效果
```python
# 计算对齐前后的域间距离
def frobenius_distance(C1, C2):
    return np.linalg.norm(C1 - C2, 'fro')

# 对齐前
C_source_mean = np.mean(C_source, axis=0)
C_target_mean = np.mean(C_target, axis=0)
dist_before = frobenius_distance(C_source_mean, C_target_mean)

# 对齐后
C_target_aligned = alignment.transform(C_target)
C_target_aligned_mean = np.mean(C_target_aligned, axis=0)
dist_after = frobenius_distance(C_source_mean, C_target_aligned_mean)

print(f"Distance before alignment: {dist_before:.3f}")
print(f"Distance after alignment: {dist_after:.3f}")
print(f"Reduction: {(dist_before - dist_after) / dist_before * 100:.1f}%")

# 对齐后距离应该减小
if dist_after >= dist_before:
    print("WARNING: Alignment increased distance! Check implementation.")
```

#### 2. 协方差矩阵可视化
```python
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 源域
sns.heatmap(C_source_mean, ax=axes[0], cmap='RdBu_r', center=0)
axes[0].set_title('Source Domain')

# 目标域（对齐前）
sns.heatmap(C_target_mean, ax=axes[1], cmap='RdBu_r', center=0)
axes[1].set_title('Target Domain (Before)')

# 目标域（对齐后）
sns.heatmap(C_target_aligned_mean, ax=axes[2], cmap='RdBu_r', center=0)
axes[2].set_title('Target Domain (After)')

plt.tight_layout()
plt.savefig('results/figures/alignment_effect.pdf')
plt.show()

# 对齐后应该更接近源域
```

#### 3. 性能提升
```python
# 无对齐
acc_no_align = evaluate_cross_subject(method='none')

# EA
acc_ea = evaluate_cross_subject(method='ea')

# RA
acc_ra = evaluate_cross_subject(method='ra')

print(f"No alignment: {acc_no_align:.3f}")
print(f"EA: {acc_ea:.3f} (+{(acc_ea - acc_no_align)*100:.1f}%)")
print(f"RA: {acc_ra:.3f} (+{(acc_ra - acc_no_align)*100:.1f}%)")

# EA应该比无对齐提升10-15%
# RA应该比EA提升3-5%
if acc_ea < acc_no_align + 0.10:
    print("WARNING: EA improvement too small! Check implementation.")

if acc_ra < acc_ea + 0.03:
    print("WARNING: RA improvement too small! Check Riemannian mean.")
```

---

## 🐛 Part 3: 常见问题和解决方案

### 问题1：准确率接近随机（25%）

**可能原因**：
1. 数据加载错误（标签错位）
2. 预处理错误（滤波参数不对）
3. 特征提取错误（CSP实现有bug）

**调试步骤**：
```python
# 1. 检查标签分布
print("Label distribution:", np.bincount(y_train))
# 应该是 [72, 72, 72, 72]

# 2. 可视化原始信号
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(X_train[i, 10, :])  # 通道10 (Cz)
    plt.title(f'Trial {i}, Label {y_train[i]}')
plt.tight_layout()
plt.show()
# 应该看到mu/beta节律

# 3. 检查CSP特征
print("Feature range:", features_train.min(), features_train.max())
print("Feature mean:", features_train.mean(axis=0))
print("Feature std:", features_train.std(axis=0))
# 特征应该有合理的范围和方差

# 4. 先在within-subject上测试
# 应该>70%，如果不是，说明基础实现有问题
```

### 问题2：对齐后性能下降

**可能原因**：
1. 对齐方向错了（应该是目标→源，不是源→目标）
2. 协方差矩阵计算错误（不是正定的）
3. 矩阵平方根计算错误

**调试步骤**：
```python
# 1. 检查对齐公式
# 正确：C_t_aligned = C_s^{1/2} @ C_t_mean^{-1/2} @ C_t @ C_t_mean^{-1/2} @ C_s^{1/2}
# 错误：C_t_aligned = C_t^{1/2} @ C_s_mean^{-1/2} @ C_s @ ...

# 2. 检查特征值
eigvals = np.linalg.eigvalsh(C_target_mean)
print("Eigenvalues:", eigvals)
# 应该都>0，如果有负值或接近0，加正则化

# 3. 可视化对齐前后的协方差矩阵
# 对齐后应该更接近源域，不是更远

# 4. 检查矩阵平方根
C_sqrt = matrix_sqrt(C_target_mean)
C_reconstructed = C_sqrt @ C_sqrt
print("Reconstruction error:", np.linalg.norm(C_reconstructed - C_target_mean))
# 应该接近0
```

### 问题3：不同被试性能差异巨大

**这是正常的！**

**原因**：
- 某些被试信号质量好（高信噪比）
- 某些被试MI能力强（模式明显）
- 某些被试对之间更相似

**处理**：
```python
# 报告平均值和标准差
print(f"Mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

# 分析哪些被试对容易迁移
transfer_matrix = compute_transfer_matrix()
plt.figure(figsize=(10, 8))
sns.heatmap(transfer_matrix, annot=True, fmt='.2f')
plt.title('Cross-Subject Transfer Performance')
plt.show()

# 这就是你的研究动机
```

### 问题4：训练时间太长

**优化方法**：
```python
# 1. 减少trial数（用前100个trial测试）
X_train_small = X_train[:100]
y_train_small = y_train[:100]

# 2. 减少被试数（先在3个被试上测试）
subjects = [0, 4, 8]  # A01, A05, A09

# 3. 使用GPU（如果用深度学习）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4. 并行化（多个被试并行训练）
from joblib import Parallel, delayed

results = Parallel(n_jobs=4)(
    delayed(train_and_evaluate)(subject)
    for subject in range(9)
)
```

**注意**：
- 最终结果必须用全部数据
- 但调试时可以用小数据集

### 问题5：CKA计算结果异常

**可能原因**：
1. 特征矩阵维度不匹配
2. 中心化错误
3. 数值不稳定

**调试步骤**：
```python
# 1. 检查输入
print(f"X shape: {X.shape}")  # (n_samples, n_features)
print(f"Y shape: {Y.shape}")  # (n_samples, n_features)
# 注意：n_samples必须相同！

# 2. 检查中心化
K = X @ X.T
H = np.eye(n) - np.ones((n, n)) / n
K_c = H @ K @ H
print(f"K_c trace: {np.trace(K_c)}")  # 应该接近0

# 3. 检查数值稳定性
print(f"K_c norm: {np.linalg.norm(K_c, 'fro')}")
# 如果太大或太小，可能有数值问题

# 4. 测试已知案例
# CKA(X, X) 应该 = 1
cka_self = cka(X, X)
print(f"CKA(X, X) = {cka_self}")
assert np.isclose(cka_self, 1.0), "CKA self-similarity should be 1!"
```

### 问题6：内存不足

**优化方法**：
```python
# 1. 分批处理
batch_size = 32
for i in range(0, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    # 处理batch

# 2. 使用生成器
def data_generator(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# 3. 降低精度
X = X.astype(np.float32)  # 从float64降到float32

# 4. 删除不需要的变量
del large_variable
import gc
gc.collect()
```

---

## 📋 Part 4: 完整验证流程

### 运行这个脚本来验证你的实现

```python
#!/usr/bin/env python
"""
完整验证脚本
"""

def validate_implementation():
    """验证所有组件"""

    print("=" * 50)
    print("验证数据加载...")
    print("=" * 50)

    # 1. 数据加载
    X_train, y_train, X_test, y_test = load_data(subject=1)

    assert X_train.shape == (288, 22, 750), "Wrong train shape!"
    assert y_train.shape == (288,), "Wrong label shape!"
    assert set(y_train) == {0, 1, 2, 3}, "Wrong label values!"

    print("✓ 数据加载正确")

    print("\n" + "=" * 50)
    print("验证CSP...")
    print("=" * 50)

    # 2. CSP
    csp = CSP(n_components=6)
    features_train = csp.fit_transform(X_train, y_train)

    assert features_train.shape == (288, 24), "Wrong feature shape!"
    assert not np.isnan(features_train).any(), "Features contain NaN!"

    print("✓ CSP正确")

    print("\n" + "=" * 50)
    print("验证LDA...")
    print("=" * 50)

    # 3. LDA
    lda = LDA(solver='lsqr', shrinkage='auto')
    lda.fit(features_train, y_train)

    train_acc = lda.score(features_train, y_train)
    assert train_acc > 0.7, f"Train accuracy too low: {train_acc}"

    features_test = csp.transform(X_test)
    test_acc = lda.score(features_test, y_test)
    assert test_acc > 0.6, f"Test accuracy too low: {test_acc}"

    print(f"✓ LDA正确 (train: {train_acc:.3f}, test: {test_acc:.3f})")

    print("\n" + "=" * 50)
    print("验证对齐...")
    print("=" * 50)

    # 4. 对齐
    C_train = compute_covariances(X_train)
    C_test = compute_covariances(X_test)

    # EA
    ea = EuclideanAlignment()
    ea.fit(C_train)
    C_test_aligned = ea.transform(C_test)

    assert C_test_aligned.shape == C_test.shape, "Wrong aligned shape!"
    assert not np.isnan(C_test_aligned).any(), "Aligned contains NaN!"

    print("✓ 对齐正确")

    print("\n" + "=" * 50)
    print("验证CKA...")
    print("=" * 50)

    # 5. CKA
    features_1 = csp.transform(X_test[:100])
    features_2 = csp.transform(X_test[100:200])

    cka_value = cka(features_1, features_2)
    assert 0 <= cka_value <= 1, f"CKA out of range: {cka_value}"

    cka_self = cka(features_1, features_1)
    assert np.isclose(cka_self, 1.0, atol=1e-3), f"CKA self != 1: {cka_self}"

    print(f"✓ CKA正确 (value: {cka_value:.3f})")

    print("\n" + "=" * 50)
    print("所有验证通过！")
    print("=" * 50)

if __name__ == '__main__':
    validate_implementation()
```

---

## 🎯 验证通过标准

运行验证脚本后，应该看到：

```
==================================================
验证数据加载...
==================================================
✓ 数据加载正确

==================================================
验证CSP...
==================================================
✓ CSP正确

==================================================
验证LDA...
==================================================
✓ LDA正确 (train: 0.823, test: 0.745)

==================================================
验证对齐...
==================================================
✓ 对齐正确

==================================================
验证CKA...
==================================================
✓ CKA正确 (value: 0.567)

==================================================
所有验证通过！
==================================================
```

如果所有检查都通过，说明你的实现是正确的，可以进入下一阶段！

---

## 📞 需要帮助？

如果遇到无法解决的问题：
1. 先运行完整验证脚本
2. 记录错误信息和堆栈跟踪
3. 检查是否符合文档要求
4. 呼叫我进行 review
