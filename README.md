# 低光照图像增强课程项目

本项目实现了一个基于 Retinex 思想的低光照图像增强实验系统，主体方法比较了 `双边滤波` 与 `引导滤波` 两种照明图平滑方式，并在增强结果之后增加了一个 `线性色彩校正` 步骤，用于缓解颜色漂移。

## 1. 项目内容

当前主流程如下：

1. 从输入图像估计初始照明图 `T0`
2. 使用双边滤波或引导滤波对照明图进行边缘保持平滑
3. 使用增强公式恢复亮度

```text
J = I / max(T, Tmin)^gamma
```

4. 对增强结果执行线性色彩校正  

```text
a' = (1 - alpha) * a_enhanced + alpha * a_input
b' = (1 - alpha) * b_enhanced + alpha * b_input
alpha = alpha_max * L_input / 100
```

5. 保存中间结果、增强结果、线性色彩校正结果和对比图
6. 计算增强结果与 Ground Truth 之间的评价指标

当前支持的方法：

- `bilateral`
- `guided`

当前使用的评价指标：

- `PSNR`
- `SSIM`
- `MAE`
- `MSE`
- `LPIPS`
- `delta_ab`

## 2. 环境与依赖

请优先使用项目根目录下的 `.venv` Python 环境。

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

检查当前 Python 是否为项目环境：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

## 3. 数据目录

项目默认读取以下目录中的训练样本：

```text
samples/
└── Train/
    ├── Input/
    └── GT/
```

程序通过文件名一一匹配 `Input` 与 `GT` 图像，例如：

- `samples/Train/Input/00001.png`
- `samples/Train/GT/00001.png`

## 4. 运行主程序

在项目根目录下运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

主程序会：

1. 遍历 `samples/Train/Input`
2. 查找对应的 `GT`
3. 分别执行双边滤波与引导滤波增强
4. 计算增强结果和线性色彩校正结果的指标
5. 保存结果到 `results/`

## 5. 输出结果

主程序输出位于 `results/`：

```text
results/
├── bilateral/
│   ├── t0/
│   ├── smooth/
│   ├── enhanced/
│   ├── comparisons/
│   ├── linear_corrected/
│   ├── linear_corrected_comparisons/
│   ├── metrics.csv
│   └── linear_corrected_metrics.csv
├── guided/
│   ├── t0/
│   ├── smooth/
│   ├── enhanced/
│   ├── comparisons/
│   ├── linear_corrected/
│   ├── linear_corrected_comparisons/
│   ├── metrics.csv
│   └── linear_corrected_metrics.csv
├── comparison_summary.csv
└── linear_corrected_comparison_summary.csv
```

说明：

- `t0/`：初始照明图
- `smooth/`：平滑后的照明图
- `enhanced/`：增强结果
- `comparisons/`：`Input / Enhanced / GT` 拼接图
- `linear_corrected/`：线性色彩校正结果
- `linear_corrected_comparisons/`：`Input / Linear Corrected / GT` 拼接图
- `metrics.csv`：增强结果的逐图像与平均指标
- `linear_corrected_metrics.csv`：线性色彩校正结果的逐图像与平均指标

## 6. 当前默认参数

当前默认参数定义在 [main.py](/C:/Users/50307/Desktop/图像处理基础/project_1/main.py)：

- 双边滤波
  - `BILATERAL_DIAMETER = 15`
  - `BILATERAL_SIGMA_COLOR = 0.1`
  - `BILATERAL_SIGMA_SPACE = 15`
- 引导滤波
  - `GUIDED_RADIUS = 8`
  - `GUIDED_EPS = 1e-3`
- 增强参数
  - `TMIN = 0.1`
  - `GAMMA = 0.8`
- 线性色彩校正
  - `LINEAR_ALPHA_MAX = 0.5`

## 7. 核心代码文件

- [main.py](/C:/Users/50307/Desktop/图像处理基础/project_1/main.py)
  - 主程序入口，负责批量处理、保存结果和输出指标
- [pipeline.py](/C:/Users/50307/Desktop/图像处理基础/project_1/pipeline.py)
  - 实现 `T0` 估计、双边滤波、引导滤波、增强恢复和线性色彩校正
- [metrics.py](/C:/Users/50307/Desktop/图像处理基础/project_1/metrics.py)
  - 实现 `PSNR`、`SSIM`、`MAE`、`MSE`、`LPIPS` 和 `delta_ab`
- [utils.py](/C:/Users/50307/Desktop/图像处理基础/project_1/utils.py)
  - 实现图像读写、Lab 转换、色差热力图、结果目录和 CSV 保存

## 8. 报告与实验脚本

报告与补充实验文件集中在 `report_workspace/`：

```text
report_workspace/
├── assets/
├── latex/
└── scripts/
```

其中：

- 报告源码：[report_workspace/latex/report.tex](/C:/Users/50307/Desktop/图像处理基础/project_1/report_workspace/latex/report.tex)
- 正式 PDF：[report_workspace/latex/report.pdf](/C:/Users/50307/Desktop/图像处理基础/project_1/report_workspace/latex/report.pdf)
- 参数实验脚本：[report_workspace/scripts/run_report_experiments.py](/C:/Users/50307/Desktop/图像处理基础/project_1/report_workspace/scripts/run_report_experiments.py)
- 线性色彩校正分析脚本：[report_workspace/scripts/run_linear_color_correction_analysis.py](/C:/Users/50307/Desktop/图像处理基础/project_1/report_workspace/scripts/run_linear_color_correction_analysis.py)

重跑补充实验：

```powershell
.\.venv\Scripts\python.exe report_workspace\scripts\run_report_experiments.py
.\.venv\Scripts\python.exe report_workspace\scripts\run_linear_color_correction_analysis.py
```

重新编译报告：

```powershell
cd report_workspace\latex
xelatex -interaction=nonstopmode report.tex
bibtex report
xelatex -interaction=nonstopmode report.tex
xelatex -interaction=nonstopmode report.tex
```

## 9. 方法特点

当前版本的特点如下：

- 结构清晰，便于展示“照明估计 + 平滑 + 增强 + 颜色校正”的完整流程
- 主体方法和参数具有较好的可解释性
- 同时支持双边滤波与引导滤波，便于定量和定性比较
- 输出中间结果完整，方便撰写课程报告
- 在线性色彩校正阶段保留增强亮度结构，并利用原图色度信息抑制颜色漂移
