# 01 - 数据准备与预处理

## 📊 Part 1: 数据集下载

### BCI Competition IV Dataset 2a

**基本信息**：
- **任务**：4类运动想象（左手、右手、双脚、舌头）
- **被试**：9人（A01-A09）
- **通道**：22个EEG通道 + 3个EOG通道
- **采样率**：250 Hz
- **Trial结构**：
  - t=0s: 提示音
  - t=2s: 提示符出现（箭头指示任务）
  - t=2-6s: 运动想象期（4秒）
  - t=6s: trial结束
- **数据量**：每个被试288个trial（训练集）+ 288个trial（测试集）

**下载地址**：
- 官方网站：http://www.bbci.de/competition/iv/
- 直接下载：http://www.bbci.de/competition/iv/desc_2a.pdf（数据描述）
- 数据文件：http://www.bbci.de/competition/iv/download/（需要注册）

**文件格式**：
- `.gdf` 格式（GDF = General Data Format for Biosignals）
- 需要用 `mne` 或 `scipy` 读取

**替代方案**（如果官网下载困难）：
- MOABB库自动下载：`from moabb.datasets import BNCI2014001`
- 这个库会自动处理下载和预处理

---

## 🔧 Part 2: 数据预处理流程

### 标准预处理pipeline（参考BCI Competition IV获奖方法）

#### Step 1: 读取原始数据
```
输入：.gdf文件
输出：原始EEG信号 (n_channels × n_samples)

工具：
- Python: mne.io.read_raw_gdf()
- 或 MOABB: dataset.get_data(subjects=[1])
```

#### Step 2: 通道选择
```
选择22个EEG通道，去掉3个EOG通道

通道列表：
['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
 'P2', 'POz']

关键通道（运动想象）：
- C3, C4（左右运动皮层）
- Cz（中央）
- CP3, CP4（感觉运动区）
```

#### Step 3: 带通滤波
```
频段：8-30 Hz（覆盖 mu 和 beta 节律）

参数：
- 滤波器类型：Butterworth 5阶
- 或 FIR滤波器（mne默认）

原理：
- mu节律（8-12 Hz）：运动想象时在对侧半球抑制
- beta节律（13-30 Hz）：运动准备和执行相关

代码示例（伪代码）：
raw.filter(l_freq=8, h_freq=30, method='iir',
           iir_params={'order': 5, 'ftype': 'butter'})
```

#### Step 4: Epoch提取
```
时间窗口：t=3-6s（运动想象的稳定期）
- 避开t=2-3s（反应时间）
- 使用t=3-6s（3秒，750个采样点@250Hz）

输出：
- X: (n_trials, n_channels, n_samples) = (288, 22, 750)
- y: (n_trials,) = (288,) 标签 [0,1,2,3]
```

#### Step 5: 基线校正（可选）
```
方法：减去trial开始前的平均值
时间窗口：t=-0.5-0s（提示音之前）

目的：去除直流偏移和慢漂移
```

---

## ✅ Part 3: 数据验证

### 在开始复现baseline之前，先验证数据质量

#### 验证1：信号质量检查
```
检查项：
1. 信号幅度范围：应该在 ±100 μV 之内
2. 是否有坏通道：某些通道全是噪声
3. 是否有伪迹：眼动、肌电等

可视化：
- 画几个trial的原始信号
- 画功率谱密度（PSD）：应该看到 mu/beta 峰
```

#### 验证2：类别平衡检查
```
检查每个类别的trial数：
- 应该是均衡的（每类72个trial）
- 如果不平衡，后续需要加权

代码示例（伪代码）：
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
# 期望输出：{0: 72, 1: 72, 2: 72, 3: 72}
```

#### 验证3：被试间差异可视化
```
目的：初步观察跨被试变异性

方法：
1. 计算每个被试的平均协方差矩阵
2. 可视化协方差矩阵（热图）
3. 计算被试间的Frobenius距离

预期：
- 不同被试的协方差矩阵应该有明显差异
- 这就是为什么需要跨被试对齐
```

---

## 📝 实现建议

### 数据加载模块结构
```python
# src/data/loader.py
class BCIDataLoader:
    def __init__(self, data_path, subjects):
        """初始化数据加载器"""
        pass

    def load_subject(self, subject_id):
        """加载单个被试数据"""
        pass

    def get_train_test_split(self, subject_id):
        """获取训练/测试集"""
        pass

# src/data/preprocessing.py
class Preprocessor:
    def __init__(self, l_freq=8, h_freq=30, tmin=3.0, tmax=6.0):
        """初始化预处理器"""
        pass

    def filter_data(self, raw):
        """带通滤波"""
        pass

    def extract_epochs(self, raw, events):
        """提取epoch"""
        pass

    def compute_covariance(self, epochs):
        """计算协方差矩阵"""
        pass
```

### 配置文件
```yaml
# configs/data_config.yaml
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
    tmin: 3.0
    tmax: 6.0
    baseline: null

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
```

---

## 🎯 完成标准

数据准备完成后，你应该有：

- ✅ 数据已下载到 `data/raw/`
- ✅ 预处理代码已实现
- ✅ 数据验证通过（信号质量、类别平衡、被试差异）
- ✅ 预处理后的数据保存到 `data/processed/`
- ✅ 数据加载和预处理可以复现

---

## 🔍 常见问题

### Q1: MOABB下载速度慢怎么办？
**A**: 可以手动从官网下载，然后放到MOABB的缓存目录：
```bash
~/.local/share/moabb/
```

### Q2: 协方差矩阵不是正定的怎么办？
**A**: 添加正则化：
```python
C = C + 1e-6 * np.eye(n_channels)
```

### Q3: 不同被试的信号幅度差异很大怎么办？
**A**: 这是正常的，这就是为什么需要对齐。可以先做归一化：
```python
X = X / np.std(X)  # 标准化
```

---

## 📚 参考资料

- BCI Competition IV 官方文档：http://www.bbci.de/competition/iv/desc_2a.pdf
- MNE-Python 教程：https://mne.tools/stable/auto_tutorials/index.html
- MOABB 文档：https://moabb.neurotechx.com/docs/
- PyRiemann 教程：https://pyriemann.readthedocs.io/
