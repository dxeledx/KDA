# 07 - 代码组织建议

本文档提供项目代码的组织结构、依赖包和配置文件建议。

---

## 📂 Part 1: 目录结构

```
DCA-BGF/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据（.gdf文件）
│   │   ├── A01T.gdf
│   │   ├── A01E.gdf
│   │   └── ...
│   ├── processed/                 # 预处理后的数据（.npy或.pkl）
│   │   ├── A01_train.npy
│   │   ├── A01_test.npy
│   │   └── ...
│   └── README.md                  # 数据说明
│
├── src/                           # 源代码
│   ├── __init__.py
│   │
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── loader.py              # 数据加载
│   │   ├── preprocessing.py       # 预处理
│   │   └── utils.py               # 工具函数
│   │
│   ├── features/                  # 特征提取模块
│   │   ├── __init__.py
│   │   ├── csp.py                 # CSP实现
│   │   ├── covariance.py          # 协方差矩阵计算
│   │   └── utils.py
│   │
│   ├── alignment/                 # 对齐方法模块
│   │   ├── __init__.py
│   │   ├── euclidean.py           # EA实现
│   │   ├── riemannian.py          # RA实现
│   │   └── coral.py               # CORAL实现
│   │
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   ├── classifiers.py         # LDA, SVM等
│   │   ├── eegnet.py              # EEGNet实现
│   │   └── otta.py                # OTTA实现
│   │
│   ├── evaluation/                # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py             # 评估指标
│   │   ├── statistics.py          # 统计检验
│   │   └── visualization.py       # 可视化
│   │
│   └── utils/                     # 通用工具
│       ├── __init__.py
│       ├── config.py              # 配置管理
│       └── logger.py              # 日志
│
├── experiments/                   # 实验脚本
│   ├── baseline_csp_lda.py        # Baseline 1
│   ├── baseline_ea.py             # Baseline 2
│   ├── baseline_ra.py             # Baseline 3
│   ├── baseline_coral.py          # Baseline 4
│   ├── baseline_otta.py           # Baseline 5
│   └── verify_phenomenon.py       # 验证表征-行为不一致
│
├── notebooks/                     # Jupyter notebooks（探索性分析）
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_phenomenon_visualization.ipynb
│
├── results/                       # 实验结果
│   ├── baselines/                 # Baseline结果
│   │   ├── csp_lda/
│   │   │   ├── metrics.json
│   │   │   └── predictions.npy
│   │   ├── ea/
│   │   └── ra/
│   ├── figures/                   # 图表
│   │   ├── rep_beh_scatter.pdf
│   │   ├── covariance_heatmaps.pdf
│   │   └── transfer_matrix.pdf
│   └── tables/                    # 表格
│       └── baseline_comparison.csv
│
├── configs/                       # 配置文件
│   ├── data_config.yaml           # 数据配置
│   ├── model_config.yaml          # 模型配置
│   └── experiment_config.yaml     # 实验配置
│
├── tests/                         # 单元测试
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_alignment.py
│   └── test_evaluation.py
│
├── docs/                          # 文档
│   ├── stage1/                    # 阶段一文档
│   │   ├── 00-overview.md
│   │   ├── 01-data-preparation.md
│   │   ├── 02-baselines-traditional.md
│   │   ├── 04-phenomenon-verification.md
│   │   ├── 06-debugging-guide.md
│   │   └── 07-code-organization.md
│   └── API.md                     # API文档
│
├── scripts/                       # 辅助脚本
│   ├── download_data.sh           # 下载数据
│   ├── preprocess_all.py          # 批量预处理
│   └── run_all_baselines.sh       # 运行所有baseline
│
├── requirements.txt               # Python依赖
├── setup.py                       # 安装脚本
├── README.md                      # 项目说明
├── .gitignore                     # Git忽略文件
└── LICENSE                        # 许可证
```

---

## 📦 Part 2: 依赖包清单

### requirements.txt
```txt
# 核心依赖
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0

# EEG处理
mne>=1.0.0
pyriemann>=0.3.0
moabb>=0.4.0

# 深度学习（如果用CORAL/OTTA）
torch>=1.10.0
torchvision>=0.11.0

# 可视化
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# 数据处理
pandas>=1.3.0
h5py>=3.0.0

# 配置管理
pyyaml>=5.4.0
hydra-core>=1.1.0

# 实验跟踪（可选）
wandb>=0.12.0
tensorboard>=2.8.0

# 工具
tqdm>=4.62.0
joblib>=1.1.0

# 测试
pytest>=7.0.0
pytest-cov>=3.0.0
```

### 安装命令
```bash
# 创建虚拟环境
conda create -n dca-bgf python=3.9
conda activate dca-bgf

# 安装依赖
pip install -r requirements.txt

# 或者用conda
conda install numpy scipy scikit-learn matplotlib seaborn pandas
conda install -c conda-forge mne pyriemann
pip install moabb torch torchvision
```

---

## ⚙️ Part 3: 配置文件

### configs/data_config.yaml
```yaml
dataset:
  name: "BCICIV2a"
  path: "data/raw"
  subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]

preprocessing:
  filter:
    l_freq: 8
    h_freq: 30
    method: "iir"
    order: 5

  epoch:
    tmin: 3.0  # 运动想象开始后1秒
    tmax: 6.0  # 运动想象结束
    baseline: null  # 不用基线校正

  channels:
    - "Fz"
    - "FC3"
    - "FC1"
    - "FCz"
    - "FC2"
    - "FC4"
    - "C5"
    - "C3"
    - "C1"
    - "Cz"
    - "C2"
    - "C4"
    - "C6"
    - "CP3"
    - "CP1"
    - "CPz"
    - "CP2"
    - "CP4"
    - "P1"
    - "Pz"
    - "P2"
    - "POz"

output:
  processed_dir: "data/processed"
  format: "numpy"  # 或 "pickle", "hdf5"
```

### configs/model_config.yaml
```yaml
csp:
  n_components: 6  # 前3个+后3个
  reg: 0.1  # 正则化参数
  log: true  # 对方差取对数

lda:
  solver: "lsqr"
  shrinkage: "auto"

ea:
  method: "euclidean"

ra:
  method: "riemann"
  metric: "riemann"

coral:
  lambda: 0.5  # CORAL损失权重
  learning_rate: 0.001
  epochs: 50

otta:
  learning_rate: 0.0001
  update_bn_only: true
```

### configs/experiment_config.yaml
```yaml
experiment:
  name: "baseline_comparison"
  seed: 42
  device: "cuda"  # 或 "cpu"

evaluation:
  protocol: "loso"  # Leave-One-Subject-Out
  metrics:
    - "accuracy"
    - "kappa"
    - "f1_macro"

  save_predictions: true
  save_features: true

logging:
  level: "INFO"
  save_dir: "results/logs"

  wandb:
    enabled: false
    project: "dca-bgf"
    entity: "your-username"
```

---

## 🔧 Part 4: 核心模块实现模板

### src/data/loader.py
```python
"""数据加载模块"""
import numpy as np
from pathlib import Path
from typing import Tuple, List

class BCIDataLoader:
    """BCI Competition IV Dataset 2a 数据加载器"""

    def __init__(self, data_path: str, subjects: List[int]):
        """
        Args:
            data_path: 数据目录路径
            subjects: 被试ID列表 [1, 2, ..., 9]
        """
        self.data_path = Path(data_path)
        self.subjects = subjects

    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个被试数据

        Args:
            subject_id: 被试ID (1-9)

        Returns:
            X: (n_trials, n_channels, n_samples)
            y: (n_trials,)
        """
        # 实现数据加载逻辑
        pass

    def get_train_test_split(self, subject_id: int) -> Tuple:
        """
        获取训练/测试集

        Returns:
            X_train, y_train, X_test, y_test
        """
        pass
```

### src/features/csp.py
```python
"""CSP特征提取"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns"""

    def __init__(self, n_components: int = 6, reg: float = 0.1):
        """
        Args:
            n_components: 特征数量（前n/2个+后n/2个）
            reg: 正则化参数
        """
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练CSP滤波器

        Args:
            X: (n_trials, n_channels, n_samples)
            y: (n_trials,)
        """
        # 实现CSP训练
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        提取CSP特征

        Args:
            X: (n_trials, n_channels, n_samples)

        Returns:
            features: (n_trials, n_components)
        """
        # 实现特征提取
        pass
```

### src/alignment/euclidean.py
```python
"""Euclidean Alignment"""
import numpy as np

class EuclideanAlignment:
    """欧几里得对齐"""

    def __init__(self):
        self.C_ref = None  # 参考协方差矩阵

    def fit(self, C_source: np.ndarray):
        """
        拟合参考协方差矩阵

        Args:
            C_source: (n_trials, n_channels, n_channels)
        """
        self.C_ref = np.mean(C_source, axis=0)

    def transform(self, C_target: np.ndarray) -> np.ndarray:
        """
        对齐目标域协方差矩阵

        Args:
            C_target: (n_trials, n_channels, n_channels)

        Returns:
            C_aligned: (n_trials, n_channels, n_channels)
        """
        # 实现对齐
        pass
```

### src/evaluation/metrics.py
```python
"""评估指标"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算所有评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        metrics: 指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }
    return metrics

def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    计算CKA相似度

    Args:
        X: (n_samples, n_features_x)
        Y: (n_samples, n_features_y)

    Returns:
        cka_value: CKA ∈ [0, 1]
    """
    # 实现CKA
    pass
```

---

## 🧪 Part 5: 实验脚本模板

### experiments/baseline_csp_lda.py
```python
"""Baseline 1: CSP + LDA"""
import numpy as np
from pathlib import Path
import yaml
import logging

from src.data.loader import BCIDataLoader
from src.data.preprocessing import Preprocessor
from src.features.csp import CSP
from src.models.classifiers import LDA
from src.evaluation.metrics import compute_metrics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 加载配置
    with open('configs/data_config.yaml') as f:
        data_config = yaml.safe_load(f)

    with open('configs/model_config.yaml') as f:
        model_config = yaml.safe_load(f)

    # 初始化
    loader = BCIDataLoader(
        data_path=data_config['dataset']['path'],
        subjects=data_config['dataset']['subjects']
    )

    preprocessor = Preprocessor(**data_config['preprocessing'])

    # LOSO评估
    results = []

    for test_subject in range(1, 10):
        logger.info(f"Testing on subject {test_subject}")

        # 加载数据
        X_train, y_train = [], []
        for train_subject in range(1, 10):
            if train_subject != test_subject:
                X, y = loader.load_subject(train_subject)
                X_train.append(X)
                y_train.append(y)

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        X_test, y_test = loader.load_subject(test_subject)

        # 预处理
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        # 训练CSP
        csp = CSP(**model_config['csp'])
        features_train = csp.fit_transform(X_train, y_train)
        features_test = csp.transform(X_test)

        # 训练LDA
        lda = LDA(**model_config['lda'])
        lda.fit(features_train, y_train)

        # 预测
        y_pred = lda.predict(features_test)

        # 评估
        metrics = compute_metrics(y_test, y_pred)
        results.append(metrics)

        logger.info(f"Subject {test_subject}: {metrics}")

    # 汇总结果
    mean_metrics = {
        key: np.mean([r[key] for r in results])
        for key in results[0].keys()
    }

    logger.info(f"Mean metrics: {mean_metrics}")

    # 保存结果
    save_dir = Path('results/baselines/csp_lda')
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / 'results.npy', results)

if __name__ == '__main__':
    main()
```

---

## 🧪 Part 6: 单元测试模板

### tests/test_features.py
```python
"""测试特征提取模块"""
import pytest
import numpy as np
from src.features.csp import CSP

def test_csp_shape():
    """测试CSP输出形状"""
    # 生成模拟数据
    n_trials, n_channels, n_samples = 100, 22, 750
    X = np.random.randn(n_trials, n_channels, n_samples)
    y = np.random.randint(0, 4, n_trials)

    # 训练CSP
    csp = CSP(n_components=6)
    features = csp.fit_transform(X, y)

    # 检查形状
    assert features.shape == (n_trials, 24), f"Wrong shape: {features.shape}"

def test_csp_no_nan():
    """测试CSP输出无NaN"""
    X = np.random.randn(100, 22, 750)
    y = np.random.randint(0, 4, 100)

    csp = CSP(n_components=6)
    features = csp.fit_transform(X, y)

    assert not np.isnan(features).any(), "Features contain NaN!"

def test_csp_deterministic():
    """测试CSP的确定性"""
    X = np.random.randn(100, 22, 750)
    y = np.random.randint(0, 4, 100)

    csp1 = CSP(n_components=6)
    features1 = csp1.fit_transform(X, y)

    csp2 = CSP(n_components=6)
    features2 = csp2.fit_transform(X, y)

    assert np.allclose(features1, features2), "CSP not deterministic!"
```

---

## 🚀 Part 7: 快速开始

### 1. 克隆项目并安装依赖
```bash
cd /Users/jason/workspace/code/workspace/Reserch_experiment/DCA-BGF
conda create -n dca-bgf python=3.9
conda activate dca-bgf
pip install -r requirements.txt
```

### 2. 下载数据
```bash
python scripts/download_data.sh
```

### 3. 预处理数据
```bash
python scripts/preprocess_all.py
```

### 4. 运行baseline
```bash
python experiments/baseline_csp_lda.py
```

### 5. 运行测试
```bash
pytest tests/
```

---

## 📝 Part 8: Git配置

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# 数据
data/raw/
data/processed/
*.gdf
*.npy
*.pkl
*.h5

# 结果
results/
*.pdf
*.png

# 配置
.env
*.local.yaml

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db
```

---

## 🎯 完成标准

代码组织完成后，你应该有：

- ✅ 清晰的目录结构
- ✅ 所有依赖已安装
- ✅ 配置文件已创建
- ✅ 核心模块已实现
- ✅ 实验脚本可运行
- ✅ 单元测试通过

---

## 📚 参考资料

- Python项目结构：https://docs.python-guide.org/writing/structure/
- Hydra配置管理：https://hydra.cc/
- Pytest测试：https://docs.pytest.org/
- Git最佳实践：https://www.conventionalcommits.org/
