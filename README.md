# 低光照图像增强作业说明

## 1. 项目功能

本项目实现了一个基于 Retinex 思路的低光照图像增强最小可运行方案，当前仅支持 **双边滤波平滑照明图** 的方法。

整体流程如下：

1. 读取 `samples/Train/Input` 中的低光照图像；
2. 使用 RGB 三通道最大值估计初始照明图 `T0`；
3. 使用双边滤波对 `T0` 进行平滑，得到照明图 `Tb`；
4. 使用公式

```text
J = I / max(Tb, Tmin)^gamma
```

得到增强结果；
5. 保存中间结果和定性对比图；
6. 使用 `samples/Train/GT` 作为 Ground Truth，计算增强结果与 GT 之间的 PSNR；
7. 将所有图像的 PSNR 保存到 `results/metrics.csv`。

## 2. 环境依赖

本项目要求优先使用项目根目录下的 `.venv` Python 环境。

当前主要依赖如下：

- Python 3.10
- numpy
- opencv-contrib-python
- scikit-image
- matplotlib
- pandas
- tqdm

依赖文件位于：

- [requirements.txt](C:/Users/50307/Desktop/图像处理基础/project_1/requirements.txt)

如果需要检查当前 Python 环境是否正确，可运行：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

当输出路径指向项目目录下的 `.venv\Scripts\python.exe` 时，说明环境正确。

## 3. 数据目录格式

当前数据目录格式如下：

```text
samples/
└─ Train/
   ├─ Input/
   └─ GT/
```

说明：

- `samples/Train/Input/`：存放低光照输入图像
- `samples/Train/GT/`：存放对应的 Ground Truth 图像

当前程序采用 **同名文件匹配** 的方式对齐 Input 和 GT，例如：

- `samples/Train/Input/00001.png`
- `samples/Train/GT/00001.png`

只有当 Input 和 GT 中存在同名文件时，该图像对才会被处理。

## 4. 如何运行

在项目根目录下，且已切换到 `.venv` 环境后，运行：

```powershell
python main.py
```

如果想显式指定项目环境，也可以运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

程序运行后会：

1. 自动遍历 `samples/Train/Input`
2. 查找对应的 `GT`
3. 对每张图像执行增强
4. 保存结果到 `results/`
5. 输出每张图像的 PSNR 和平均 PSNR

## 5. 输出结果在哪里

所有结果统一保存在项目根目录下的 `results/` 文件夹中，结构如下：

```text
results/
├─ t0/
├─ tb/
├─ enhanced/
├─ comparisons/
└─ metrics.csv
```

各目录含义如下：

- `results/t0/`：保存初始照明图 `T0`
- `results/tb/`：保存双边滤波后的照明图 `Tb`
- `results/enhanced/`：保存增强结果图像
- `results/comparisons/`：保存 `Input / Enhanced / GT` 三图拼接对比图
- `results/metrics.csv`：保存每张图像的 PSNR 以及平均 PSNR

## 6. PSNR 指标如何计算

本项目当前只实现 PSNR，不实现 SSIM、LPIPS。

PSNR 基于增强结果图像 `J` 与 Ground Truth 图像 `GT` 的均方误差 MSE 计算：

```text
MSE = mean((J - GT)^2)
PSNR = 10 * log10(1 / MSE)
```

说明：

- 程序内部先将图像归一化到 `[0, 1]`
- 因此这里的峰值最大值使用 `1.0`
- 如果 `MSE` 极小，说明增强结果与 GT 非常接近，PSNR 会更高

一般来说：

- PSNR 越大，说明增强结果与 GT 越接近
- PSNR 越小，说明结果与 GT 偏差越大

## 7. 当前方法的主要参数含义

当前主要参数定义在 [main.py](C:/Users/50307/Desktop/图像处理基础/project_1/main.py) 中。

### 7.1 双边滤波参数

`BILATERAL_DIAMETER`

- 含义：双边滤波邻域直径 `d`
- 作用：决定每个像素参与平滑的邻域范围
- 影响：值越大，平滑范围通常越大，计算量也会增加

`BILATERAL_SIGMA_COLOR`

- 含义：双边滤波中的值域参数 `sigmaColor`
- 作用：控制像素值差异对滤波权重的影响
- 影响：值越大，亮度差异较大的像素也更容易互相影响

`BILATERAL_SIGMA_SPACE`

- 含义：双边滤波中的空间域参数 `sigmaSpace`
- 作用：控制空间距离对滤波权重的影响
- 影响：值越大，更远的邻域像素也会参与更多平滑

### 7.2 增强公式参数

`TMIN`

- 含义：照明图下限值 `Tmin`
- 作用：防止 `Tb` 过小导致除法不稳定或局部过增强
- 影响：值太小可能导致暗区增强过强；值太大可能导致增强不足

`GAMMA`

- 含义：增强公式中的指数参数 `gamma`
- 作用：控制增强强度
- 影响：
  - 当 `gamma` 较大时，增强程度通常更强
  - 当 `gamma` 较小时，增强程度通常更温和

## 8. 当前代码文件说明

当前核心文件如下：

- [main.py](C:/Users/50307/Desktop/图像处理基础/project_1/main.py)：主程序入口，负责批量处理、保存结果、输出指标
- [pipeline.py](C:/Users/50307/Desktop/图像处理基础/project_1/pipeline.py)：实现 `T0` 估计、双边滤波和增强公式
- [metrics.py](C:/Users/50307/Desktop/图像处理基础/project_1/metrics.py)：实现 PSNR 计算
- [utils.py](C:/Users/50307/Desktop/图像处理基础/project_1/utils.py)：实现图像读写、结果目录创建、对比图拼接、CSV 保存

## 9. 当前方法特点

本项目当前实现的是一个课程作业用的基础版本，特点如下：

- 方法简单，流程清晰，便于解释
- 只使用双边滤波，不涉及更复杂的引导滤波或深度学习方法
- 适合用于展示 Retinex 思路下“照明估计 + 平滑 + 补偿增强”的基本流程

如果后续需要做实验扩展，可以继续在当前结构上增加：

- 参数对比实验
- 其他照明图平滑方法
- 更多评价指标
