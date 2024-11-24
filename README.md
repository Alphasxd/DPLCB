## 项目简介

本项目实现了在联邦学习环境下使用差分隐私算法进行多臂赌博机实验。

### 文件说明

**Utils.py**
> 包含实验中使用的工具函数，如特征提取、单次算法评估和多次算法评估。

**Fed_Lin_Plot.py**
> 用于绘制实验结果的图表，展示不同算法在不同环境下的表现。

**Fed_Lin_Bandit.py**
> 实验的主函数，定义了不同的实验环境和算法，并运行实验。

**Environments.py**
> 定义了实验中使用的不同赌博机环境，包括合成环境和半合成环境。

**Datasets.py**
> 处理数据集的类，主要用于加载和预处理 MSLR-WEB10K 数据集。

**Agents.py**
> 定义了不同的算法类，包括非隐私和差分隐私算法。

### 文件引用关系

- `Fed_Lin_Bandit.py` 引用了 `Utils.py`、`Environments.py` 和 `Agents.py` 中的函数和类。
- `Fed_Lin_Plot.py` 读取 `Fed_Lin_Bandit.py` 生成的实验结果文件，并绘制图表。
- `Environments.py` 引用了 `Datasets.py` 中的数据集处理类。
- `Utils.py` 提供了实验评估的工具函数，被 `Fed_Lin_Bandit.py` 调用。

### 运行实验

1. 确保所有依赖库已安装：
    
    ```shell
    pip install -r requirements.txt
    ```

2. 运行 Fed_Lin_Bandit.py 进行实验：

    ```shell
    python Fed_Lin_Bandit.py
    ```

3. 运行 Fed_Lin_Plot.py 绘制实验结果图表：

    ```shell
    python Fed_Lin_Plot.py
    ```

### 数据集[MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/)

数据集的处理和加载方法如下：
1. 下载 MSLR-WEB10K 数据集，并将其保存为 mslr.txt 文件。
2. 使用 Datasets.py 中的 loadTxt 方法将数据集转换为 .npz 格式，并保存处理后的数据：

    ```python
    mslrData = Datasets()
    mslrData.loadTxt('mslr.txt', 'MSLR10k')
    del mslrData
    ```

3. 在实验中使用 loadNpz 方法加载处理后的数据集：

    ```python
    dataset = Datasets()
    dataset.loadNpz('mslr')
    ```