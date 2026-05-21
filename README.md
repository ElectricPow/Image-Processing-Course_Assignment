# 低光照图像增强作业说明

## 1. 项目功能

本项目实现了一个基于 Retinex 思路的低光照图像增强最小可运行方案，当前采用“最大通道估计初始照明图 + 双边滤波平滑照明图 + 增强恢复”的流程。

整体流程如下：
1. 读取 `samples/Train/Input` 中的低光照输入图像。
2. 使用 RGB 三通道最大值估计初始照明图 `T0`。
3. 使用双边滤波对 `T0` 进行平滑，得到照明图 `Tb`。
4. 使用增强公式恢复增强结果：

```text
J = I / max(Tb, Tmin)^gamma
```

5. 保存中间结果图、增强结果图和对比图。
6. 使用 `samples/Train/GT` 作为 Ground Truth，计算增强结果与 GT 之间的评价指标。
7. 将每张图像及其平均指标保存到 `results/metrics.csv`。

当前已实现的评价指标包括：
- `PSNR`
- `SSIM`
- `MAE`
- `MSE`
- `LPIPS`

## 2. 环境依赖

本项目要求优先使用项目根目录下的 `.venv` Python 环境。

当前主要依赖如下：
- Python 3.10 及以上
- numpy
- opencv-contrib-python
- scikit-image
- matplotlib
- pandas
- tqdm
- torch
- lpips

依赖文件位于：
- [requirements.txt](C:/Users/50307/Desktop/图像处理基础/project_1/requirements.txt)

如需安装依赖，可在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需检查当前 Python 环境是否正确，可运行：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -m pip --version
```

当输出路径指向项目目录下的 `.venv\Scripts\python.exe` 时，说明环境正确。

## 3. 数据目录格式

当前数据目录结构如下：

```text
samples/
└── Train/
    ├── Input/
    └── GT/
```

说明：
- `samples/Train/Input/`：存放低光照输入图像。
- `samples/Train/GT/`：存放对应的 Ground Truth 图像。

程序采用“同名文件匹配”的方式对齐 Input 和 GT，例如：
- `samples/Train/Input/00001.png`
- `samples/Train/GT/00001.png`

只有当 Input 和 GT 中存在同名文件时，该图像对才会被处理。

## 4. 如何运行

在项目根目录下，并确保使用 `.venv` 环境后，运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

程序运行后会：
1. 自动遍历 `samples/Train/Input`。
2. 查找对应的 `GT`。
3. 对每张图像执行增强。
4. 保存结果到 `results/`。
5. 输出每张图像以及平均结果的 `PSNR`、`SSIM`、`MAE`、`MSE`、`LPIPS`。

## 5. 输出结果位置

所有结果统一保存在项目根目录下的 `results/` 文件夹中，结构如下：

```text
results/
├── t0/
├── tb/
├── enhanced/
├── comparisons/
└── metrics.csv
```

各目录含义如下：
- `results/t0/`：保存初始照明图 `T0`。
- `results/tb/`：保存双边滤波后的照明图 `Tb`。
- `results/enhanced/`：保存增强结果图像。
- `results/comparisons/`：保存 `Input / Enhanced / GT` 三图拼接对比图。
- `results/metrics.csv`：保存每张图像以及平均结果的各项评价指标。

## 6. 评价指标说明

当前项目实现了以下 5 个评价指标：

### 6.1 PSNR

PSNR 基于增强结果图像 `J` 与 Ground Truth 图像 `GT` 的均方误差 `MSE` 计算：

```text
MSE = mean((J - GT)^2)
PSNR = 10 * log10(1 / MSE)
```

说明：
- 程序内部默认先将图像归一化到 `[0, 1]`。
- 因此这里的峰值最大值取 `1.0`。
- `PSNR` 越大，说明增强结果与 GT 越接近。

### 6.2 SSIM

SSIM 使用 `scikit-image` 标准实现计算，用于衡量增强结果与 GT 在亮度、对比度和结构上的相似性。

说明：
- 当前实现通过 `skimage.metrics.structural_similarity` 计算。
- 输入图像默认已归一化到 `[0, 1]`，因此 `data_range=1.0`。
- `SSIM` 越接近 `1`，说明结构相似性越高。

### 6.3 MAE

MAE 表示逐像素绝对误差的平均值：

```text
MAE = mean(abs(J - GT))
```

说明：
- `MAE` 越小，说明增强结果与 GT 的平均偏差越小。

### 6.4 MSE

MSE 表示逐像素平方误差的平均值：

```text
MSE = mean((J - GT)^2)
```

说明：
- `MSE` 越小，说明增强结果与 GT 的误差越小。
- `PSNR` 就是基于 `MSE` 进一步计算得到的。

### 6.5 LPIPS

LPIPS 使用感知特征距离评价增强结果与 GT 的视觉感知差异，当前代码使用标准 `lpips` 库实现。

说明：
- 当前实现使用 `LPIPS(net="alex")`。
- 输入图像会从 `[0, 1]` 转换到 `[-1, 1]` 后再送入模型。
- `LPIPS` 越小，说明两张图在感知上越相近。

## 7. 当前方法的主要参数

当前主要参数定义在 [main.py](C:/Users/50307/Desktop/图像处理基础/project_1/main.py) 中。

### 7.1 双边滤波参数

`BILATERAL_DIAMETER`
- 含义：双边滤波邻域直径 `d`。
- 作用：决定每个像素参与平滑的邻域范围。

`BILATERAL_SIGMA_COLOR`
- 含义：双边滤波中的值域参数 `sigmaColor`。
- 作用：控制像素值差异对滤波权重的影响。

`BILATERAL_SIGMA_SPACE`
- 含义：双边滤波中的空间域参数 `sigmaSpace`。
- 作用：控制空间距离对滤波权重的影响。

### 7.2 增强公式参数

`TMIN`
- 含义：照明图下限值 `Tmin`。
- 作用：防止 `Tb` 过小导致除法不稳定或局部过增强。

`GAMMA`
- 含义：增强公式中的指数参数 `gamma`。
- 作用：控制增强强度。

## 8. 当前代码文件说明

当前核心文件如下：
- [main.py](C:/Users/50307/Desktop/图像处理基础/project_1/main.py)：主程序入口，负责批量处理、保存结果、输出指标。
- [pipeline.py](C:/Users/50307/Desktop/图像处理基础/project_1/pipeline.py)：实现 `T0` 估计、双边滤波和平滑增强流程。
- [metrics.py](C:/Users/50307/Desktop/图像处理基础/project_1/metrics.py)：实现 `PSNR`、`SSIM`、`MAE`、`MSE`、`LPIPS` 的计算，以及平均指标统计。
- [utils.py](C:/Users/50307/Desktop/图像处理基础/project_1/utils.py)：实现图像读写、结果目录创建、对比图拼接和 CSV 保存。

## 9. 当前方法特点

本项目当前实现的是一个课程作业用的基础版本，特点如下：
- 方法结构清晰，便于报告解释和实验展示。
- 保留了 Retinex 思路下“照明估计 + 平滑 + 补偿增强”的基本流程。
- 已支持多种常见评价指标，便于从像素误差、结构相似性和感知质量三个角度分析结果。
- 适合作为后续参数对比实验和方法扩展的基础版本。
