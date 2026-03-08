# 阶段二 - 实现指南 (Implementation Guide)

## 1. 代码结构

### 1.1 目录组织

```
DCA-BGF/
├── src/
│   ├── alignment/
│   │   ├── __init__.py
│   │   ├── base.py                    # 基础对齐接口
│   │   ├── euclidean.py               # EA实现（复用阶段一）
│   │   ├── riemannian.py              # RA实现（复用阶段一）
│   │   ├── conditional.py             # 条件对齐网络
│   │   ├── behavior_feedback.py       # 行为引导反馈
│   │   └── dca_bgf.py                 # 完整的DCA-BGF
│   ├── data/
│   │   └── loader.py                  # 数据加载（复用阶段一）
│   ├── features/
│   │   └── csp.py                     # CSP特征提取（复用阶段一）
│   ├── models/
│   │   └── classifier.py              # LDA分类器（复用阶段一）
│   ├── evaluation/
│   │   ├── metrics.py                 # 评估指标（复用阶段一）
│   │   └── visualization.py           # 可视化工具
│   └── utils/
│       ├── context.py                 # 上下文计算
│       └── monitoring.py              # 监控指标计算
├── experiments/
│   ├── dca_bgf_mvp.py                 # MVP实验脚本
│   ├── dca_bgf_full.py                # 完整实验脚本
│   └── ablation_study.py              # 消融研究脚本
├── configs/
│   ├── mvp_config.yaml                # MVP配置
│   └── full_config.yaml               # 完整配置
├── notebooks/
│   ├── 01_debug_conditional.ipynb     # 调试条件网络
│   ├── 02_debug_feedback.ipynb        # 调试反馈机制
│   └── 03_visualize_results.ipynb     # 结果可视化
└── tests/
    ├── test_conditional.py            # 条件网络单元测试
    ├── test_feedback.py               # 反馈机制单元测试
    └── test_dca_bgf.py                # 完整系统测试
```

### 1.2 模块依赖关系

```
dca_bgf.py
    ├── conditional.py
    │   ├── context.py
    │   └── base.py (EA/RA)
    ├── behavior_feedback.py
    │   └── monitoring.py
    ├── classifier.py
    └── csp.py
```

---

## 2. 核心模块实现

### 2.1 上下文计算 (context.py)

```python
# src/utils/context.py
import numpy as np
from typing import List, Dict, Optional

class ContextComputer:
    """计算上下文向量 c_t"""

    def __init__(self, source_mean: np.ndarray, context_dim: int = 3):
        """
        Args:
            source_mean: 源域特征均值 μ_src
            context_dim: 上下文维度（3=MVP, 6=完整版本）
        """
        self.source_mean = source_mean
        self.target_mean = None  # 目标域均值（在线更新）
        self.context_dim = context_dim
        self.alpha = 0.1  # EMA系数

    def compute(self, x_t: np.ndarray, history: List[Dict]) -> np.ndarray:
        """
        计算当前trial的上下文向量

        Args:
            x_t: 当前trial特征 (d,)
            history: 历史信息列表

        Returns:
            c_t: 上下文向量 (context_dim,)
        """
        # 特征1：到源域中心的距离
        d_src = np.linalg.norm(x_t - self.source_mean)

        # 特征2：到目标域中心的距离
        if self.target_mean is None:
            d_tgt = 0.0
        else:
            d_tgt = np.linalg.norm(x_t - self.target_mean)

        # 特征3：最近trial的方差
        if len(history) < 5:
            sigma_recent = 0.0
        else:
            recent_features = [h['x'] for h in history[-5:]]
            sigma_recent = np.std(recent_features)

        # MVP版本：只用几何特征
        if self.context_dim == 3:
            c_t = np.array([d_src, d_tgt, sigma_recent])
        # 完整版本：添加行为特征
        elif self.context_dim == 6:
            # 特征4：预测熵
            if len(history) == 0:
                H_t = 0.0
            else:
                H_t = history[-1].get('entropy', 0.0)

            # 特征5：平均置信度
            if len(history) < 5:
                conf_avg = 0.0
            else:
                conf_avg = np.mean([h['conf'] for h in history[-5:]])

            # 特征6：KL散度
            if len(history) < 10:
                KL_div = 0.0
            else:
                KL_div = self._compute_kl_div(history[-10:])

            c_t = np.array([d_src, d_tgt, sigma_recent, H_t, conf_avg, KL_div])
        else:
            raise ValueError(f"Unsupported context_dim: {self.context_dim}")

        # 更新目标域均值（EMA）
        if self.target_mean is None:
            self.target_mean = x_t.copy()
        else:
            self.target_mean = (1 - self.alpha) * self.target_mean + self.alpha * x_t

        return c_t

    def _compute_kl_div(self, history: List[Dict]) -> float:
        """计算类别分布的KL散度"""
        # 统计预测类别分布
        predictions = [h['y_pred'] for h in history]
        n_classes = 4  # BCI Competition IV 2a有4个类别
        class_counts = np.bincount(predictions, minlength=n_classes)
        class_dist = class_counts / len(predictions)

        # 均匀分布
        uniform_dist = np.ones(n_classes) / n_classes

        # KL散度（添加小常数避免log(0)）
        epsilon = 1e-10
        kl_div = np.sum(class_dist * np.log((class_dist + epsilon) / (uniform_dist + epsilon)))

        return kl_div

    def normalize(self, c_t: np.ndarray, stats: Dict) -> np.ndarray:
        """归一化上下文向量"""
        mean = stats['mean']
        std = stats['std']
        return (c_t - mean) / (std + 1e-8)
```

### 2.2 条件对齐网络 (conditional.py)

```python
# src/alignment/conditional.py
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class ConditionalAlignmentNetwork(nn.Module):
    """条件对齐网络：预测对齐权重 w_t"""

    def __init__(self, context_dim: int = 3, hidden_dim: int = 16):
        """
        Args:
            context_dim: 上下文维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 w_t ∈ [0, 1]
        )

    def forward(self, c_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_t: 上下文向量 (batch_size, context_dim)

        Returns:
            w_t: 对齐权重 (batch_size, 1)
        """
        return self.net(c_t)


class ConditionalAligner:
    """条件对齐器：结合条件网络和基础对齐方法"""

    def __init__(
        self,
        base_aligner,  # EA或RA对齐器
        conditional_net: ConditionalAlignmentNetwork,
        context_computer,
        device: str = 'cpu'
    ):
        self.base_aligner = base_aligner
        self.conditional_net = conditional_net.to(device)
        self.context_computer = context_computer
        self.device = device

    def align(self, x_t: np.ndarray, history: list) -> np.ndarray:
        """
        对齐单个trial

        Args:
            x_t: 当前trial特征 (d,)
            history: 历史信息

        Returns:
            x_t_aligned: 对齐后的特征 (d,)
        """
        # 1. 计算上下文
        c_t = self.context_computer.compute(x_t, history)

        # 2. 预测对齐权重
        c_t_tensor = torch.FloatTensor(c_t).unsqueeze(0).to(self.device)
        with torch.no_grad():
            w_t = self.conditional_net(c_t_tensor).item()

        # 3. 基础对齐
        x_t_base_aligned = self.base_aligner.transform(x_t.reshape(1, -1))[0]

        # 4. 条件对齐（线性插值）
        x_t_aligned = (1 - w_t) * x_t + w_t * x_t_base_aligned

        return x_t_aligned, w_t

    def train_network(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        classifier,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        lambda_smooth: float = 0.1
    ):
        """
        训练条件对齐网络

        Args:
            source_data: 源域特征 (N, d)
            source_labels: 源域标签 (N,)
            classifier: 分类器
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            lambda_smooth: 平滑损失权重
        """
        optimizer = torch.optim.Adam(self.conditional_net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        N = len(source_data)
        history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = np.random.permutation(N)

            for i in range(0, N, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_x = source_data[batch_idx]
                batch_y = source_labels[batch_idx]

                # 计算上下文和对齐权重
                batch_c = []
                batch_w = []
                batch_x_aligned = []

                for j, x_j in enumerate(batch_x):
                    # 模拟在线场景：使用之前的trial作为历史
                    hist = history[:batch_idx[j]]
                    c_j = self.context_computer.compute(x_j, hist)
                    batch_c.append(c_j)

                    # 预测对齐权重
                    c_j_tensor = torch.FloatTensor(c_j).unsqueeze(0).to(self.device)
                    w_j = self.conditional_net(c_j_tensor)
                    batch_w.append(w_j)

                    # 对齐
                    x_j_base = self.base_aligner.transform(x_j.reshape(1, -1))[0]
                    x_j_aligned = (1 - w_j.item()) * x_j + w_j.item() * x_j_base
                    batch_x_aligned.append(x_j_aligned)

                # 分类损失
                batch_x_aligned = np.array(batch_x_aligned)
                y_pred = classifier.predict_proba(batch_x_aligned)
                y_pred_tensor = torch.FloatTensor(y_pred).to(self.device)
                y_true_tensor = torch.LongTensor(batch_y).to(self.device)
                loss_cls = criterion(y_pred_tensor, y_true_tensor)

                # 平滑损失
                if len(batch_w) > 1:
                    w_tensor = torch.cat(batch_w)
                    loss_smooth = torch.mean((w_tensor[1:] - w_tensor[:-1]) ** 2)
                else:
                    loss_smooth = torch.tensor(0.0).to(self.device)

                # 总损失
                loss = loss_cls + lambda_smooth * loss_smooth

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 更新历史
                for j, x_j in enumerate(batch_x):
                    history.append({
                        'x': x_j,
                        'y_pred': np.argmax(y_pred[j]),
                        'conf': np.max(y_pred[j])
                    })

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/N:.4f}")
```

### 2.3 行为引导反馈 (behavior_feedback.py)

```python
# src/alignment/behavior_feedback.py
import numpy as np
from typing import List, Dict
from scipy.stats import linregress

class BehaviorGuidedFeedback:
    """行为引导反馈机制"""

    def __init__(
        self,
        window_size: int = 10,
        H_threshold_high: float = 1.2,
        H_threshold_low: float = 0.4,
        conf_trend_threshold: float = -0.05,
        alpha: float = 0.1,
        beta: float = 0.05,
        momentum: float = 0.7
    ):
        """
        Args:
            window_size: 滑动窗口大小
            H_threshold_high: 高熵阈值
            H_threshold_low: 低熵阈值
            conf_trend_threshold: 置信度趋势阈值
            alpha: 增加步长
            beta: 减少步长
            momentum: 动量系数
        """
        self.window_size = window_size
        self.H_threshold_high = H_threshold_high
        self.H_threshold_low = H_threshold_low
        self.conf_trend_threshold = conf_trend_threshold
        self.alpha = alpha
        self.beta = beta
        self.momentum = momentum
        self.velocity = 0.0  # 动量

    def adjust_weight(self, w_t: float, history: List[Dict]) -> float:
        """
        根据历史信息调整对齐权重

        Args:
            w_t: 当前对齐权重
            history: 历史信息列表

        Returns:
            w_t_new: 调整后的对齐权重
        """
        if len(history) < self.window_size:
            # 冷启动阶段：不调整
            return w_t

        # 计算监控指标
        metrics = self._compute_metrics(history)

        # 计算调整量
        delta_w = 0.0

        # 规则1：不确定性反馈
        if metrics['H_avg'] > self.H_threshold_high:
            delta_w += self.alpha
        elif metrics['H_avg'] < self.H_threshold_low:
            delta_w -= self.beta

        # 规则2：置信度趋势反馈
        if metrics['conf_trend'] < self.conf_trend_threshold:
            delta_w += 0.15

        # 规则3：类别偏差反馈（暂不实现，留作扩展）

        # 应用动量
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * delta_w
        w_t_new = w_t + self.velocity

        # 限制范围
        w_t_new = np.clip(w_t_new, 0.0, 1.0)

        return w_t_new

    def _compute_metrics(self, history: List[Dict]) -> Dict:
        """计算监控指标"""
        window = history[-self.window_size:]

        # 平均熵
        entropies = [h.get('entropy', 0.0) for h in window]
        H_avg = np.mean(entropies)

        # 置信度趋势
        confidences = [h['conf'] for h in window]
        if len(confidences) >= 5:
            slope, _, _, _, _ = linregress(range(len(confidences)), confidences)
            conf_trend = slope
        else:
            conf_trend = 0.0

        return {
            'H_avg': H_avg,
            'conf_trend': conf_trend
        }
```

### 2.4 完整的DCA-BGF (dca_bgf.py)

```python
# src/alignment/dca_bgf.py
import numpy as np
from typing import List, Dict, Tuple
from .conditional import ConditionalAligner
from .behavior_feedback import BehaviorGuidedFeedback

class DCABGF:
    """DCA-BGF: Dynamic Conditional Alignment with Behavior-Guided Feedback"""

    def __init__(
        self,
        conditional_aligner: ConditionalAligner,
        behavior_feedback: BehaviorGuidedFeedback,
        classifier,
        use_feedback: bool = True
    ):
        """
        Args:
            conditional_aligner: 条件对齐器
            behavior_feedback: 行为引导反馈
            classifier: 分类器
            use_feedback: 是否使用行为反馈
        """
        self.conditional_aligner = conditional_aligner
        self.behavior_feedback = behavior_feedback
        self.classifier = classifier
        self.use_feedback = use_feedback

    def predict_online(
        self,
        trial_stream: np.ndarray,
        return_details: bool = False
    ) -> np.ndarray:
        """
        在线预测

        Args:
            trial_stream: 目标域trial流 (T, d)
            return_details: 是否返回详细信息

        Returns:
            predictions: 预测标签 (T,)
            details: 详细信息（如果return_details=True）
        """
        T = len(trial_stream)
        predictions = []
        history = []
        details = {
            'w_t': [],
            'conf': [],
            'entropy': []
        }

        w_t = 0.5  # 初始对齐权重

        for t, x_t in enumerate(trial_stream):
            # 1. 条件对齐
            x_t_aligned, w_t_pred = self.conditional_aligner.align(x_t, history)

            # 2. 行为引导反馈
            if self.use_feedback and t >= 10:
                w_t = self.behavior_feedback.adjust_weight(w_t_pred, history)
                # 重新对齐
                x_t_base = self.conditional_aligner.base_aligner.transform(x_t.reshape(1, -1))[0]
                x_t_aligned = (1 - w_t) * x_t + w_t * x_t_base
            else:
                w_t = w_t_pred

            # 3. 分类
            y_pred_proba = self.classifier.predict_proba(x_t_aligned.reshape(1, -1))[0]
            y_pred = np.argmax(y_pred_proba)
            conf = np.max(y_pred_proba)

            # 4. 计算熵
            entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-10))

            # 5. 记录
            predictions.append(y_pred)
            history.append({
                'x': x_t,
                'y_pred': y_pred,
                'conf': conf,
                'entropy': entropy,
                'w': w_t
            })

            if return_details:
                details['w_t'].append(w_t)
                details['conf'].append(conf)
                details['entropy'].append(entropy)

        if return_details:
            return np.array(predictions), details
        else:
            return np.array(predictions)
```

---

## 3. 实验脚本

### 3.1 MVP实验 (dca_bgf_mvp.py)

```python
# experiments/dca_bgf_mvp.py
import numpy as np
from src.data.loader import BCIDataLoader
from src.features.csp import CSP
from src.alignment.euclidean import EuclideanAlignment
from src.alignment.conditional import ConditionalAlignmentNetwork, ConditionalAligner
from src.alignment.behavior_feedback import BehaviorGuidedFeedback
from src.alignment.dca_bgf import DCABGF
from src.models.classifier import LDAClassifier
from src.utils.context import ContextComputer
from sklearn.metrics import accuracy_score

def run_mvp_experiment(subject_id: int):
    """运行MVP实验"""
    # 1. 加载数据
    loader = BCIDataLoader(data_dir='data/processed')
    X_train, y_train = loader.load_subject(subject_id, session='train')
    X_test, y_test = loader.load_subject(subject_id, session='test')

    # 2. CSP特征提取
    csp = CSP(n_components=6)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    # 3. 训练基础分类器（在源域上）
    # 这里简化：用同一个被试的训练集作为源域
    classifier = LDAClassifier()
    classifier.fit(X_train_csp, y_train)

    # 4. 基础对齐
    ea = EuclideanAlignment()
    ea.fit(X_train_csp)

    # 5. 上下文计算器
    source_mean = np.mean(X_train_csp, axis=0)
    context_computer = ContextComputer(source_mean, context_dim=3)

    # 6. 条件对齐网络
    conditional_net = ConditionalAlignmentNetwork(context_dim=3, hidden_dim=16)
    conditional_aligner = ConditionalAligner(
        base_aligner=ea,
        conditional_net=conditional_net,
        context_computer=context_computer
    )

    # 7. 训练条件网络
    print("Training conditional network...")
    conditional_aligner.train_network(
        X_train_csp, y_train, classifier,
        epochs=50, batch_size=32, lr=1e-3
    )

    # 8. 行为引导反馈
    behavior_feedback = BehaviorGuidedFeedback(window_size=10)

    # 9. 完整的DCA-BGF
    dca_bgf = DCABGF(
        conditional_aligner=conditional_aligner,
        behavior_feedback=behavior_feedback,
        classifier=classifier,
        use_feedback=True
    )

    # 10. 在线预测
    print("Online prediction...")
    y_pred, details = dca_bgf.predict_online(X_test_csp, return_details=True)

    # 11. 评估
    acc = accuracy_score(y_test, y_pred)
    print(f"Subject {subject_id} - Accuracy: {acc:.4f}")

    return acc, details

if __name__ == '__main__':
    # 在所有被试上运行
    results = {}
    for subject_id in range(1, 10):  # BCI Competition IV 2a有9个被试
        acc, details = run_mvp_experiment(subject_id)
        results[subject_id] = {'acc': acc, 'details': details}

    # 平均准确率
    avg_acc = np.mean([r['acc'] for r in results.values()])
    print(f"\nAverage Accuracy: {avg_acc:.4f}")
```

---

## 4. 调试技巧

### 4.1 单元测试

```python
# tests/test_conditional.py
import pytest
import numpy as np
from src.alignment.conditional import ConditionalAlignmentNetwork

def test_conditional_network():
    """测试条件网络"""
    net = ConditionalAlignmentNetwork(context_dim=3, hidden_dim=16)

    # 测试前向传播
    c_t = torch.randn(5, 3)  # batch_size=5
    w_t = net(c_t)

    assert w_t.shape == (5, 1)
    assert torch.all(w_t >= 0) and torch.all(w_t <= 1)

def test_context_computer():
    """测试上下文计算"""
    source_mean = np.zeros(10)
    context_computer = ContextComputer(source_mean, context_dim=3)

    x_t = np.random.randn(10)
    history = []
    c_t = context_computer.compute(x_t, history)

    assert c_t.shape == (3,)
    assert not np.any(np.isnan(c_t))
```

### 4.2 可视化调试

```python
# notebooks/01_debug_conditional.ipynb
import matplotlib.pyplot as plt

# 可视化对齐权重
plt.figure(figsize=(12, 4))
plt.plot(details['w_t'])
plt.xlabel('Trial')
plt.ylabel('Alignment Weight')
plt.title('Evolution of Alignment Weight')
plt.grid(True)
plt.show()

# 可视化置信度
plt.figure(figsize=(12, 4))
plt.plot(details['conf'])
plt.xlabel('Trial')
plt.ylabel('Confidence')
plt.title('Prediction Confidence over Time')
plt.grid(True)
plt.show()
```

---

## 5. 常见问题

### Q1: 训练条件网络时loss不下降怎么办？

**检查清单**：
1. 数据是否正确加载
2. 上下文特征是否归一化
3. 学习率是否合适（尝试1e-4或1e-2）
4. 是否有梯度消失（检查梯度范数）

### Q2: 在线预测时程序很慢怎么办？

**优化方案**：
1. 使用EA而不是RA（更快）
2. 预计算源域统计量
3. 使用GPU加速条件网络
4. 批量处理（如果允许延迟）

### Q3: 如何快速验证实现是否正确？

**验证步骤**：
1. 在单个被试上测试（within-subject）
2. 检查对齐权重是否在[0,1]范围内
3. 检查准确率是否合理（>40%）
4. 可视化对齐权重和置信度的变化

---

## 6. 下一步

完成实现后，继续阅读：
- **05-experiment-design.md**：设计完整的实验方案
- **06-expected-results.md**：了解预期结果和失败预案

---

**文档版本**: v1.0
**最后更新**: 2026-03-05
**状态**: 实现指南完成
