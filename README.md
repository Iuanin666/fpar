# FPAR 遥感融合估算项目

## 项目概述

本项目基于 **Sentinel-1 SAR** 雷达数据，利用 **U-Net 深度学习模型** 估算植被 **FPAR**（光合有效辐射吸收比，Fraction of Absorbed Photosynthetically Active Radiation）。

**核心思路**：利用 Sentinel-1（全天候、不受云雨影响）的 VV/VH 双极化后向散射数据作为输入，以 Sentinel-2 反演的 FPAR 产品作为标签（ground truth），训练 U-Net 模型实现 SAR → FPAR 的回归预测。

## 目录结构

```
FPAR_project/
├── src/                    # 源代码
│   ├── 1_align_data.py     # 将 S2(20m) 重采样对齐至 S1(10m) 网格
│   ├── 2_dataset.py        # PyTorch Dataset：S1/S2/MODIS 配对与采样
│   ├── 3_unet.py           # U-Net 模型定义（2通道输入 → 1通道输出）
│   ├── 4_train.py          # U-Net 训练脚本（GPU/AMP/TensorBoard/EarlyStopping）
│   ├── 5_transformer_model.py # Dual-Stream Transformer 模型定义
│   ├── 6_train_transformer.py # Transformer 训练脚本
│   ├── 7_align_modis.py    # MODIS 对齐：1000m→500m 重采样 + 重投影裁剪
│   ├── 8_crossscale_model.py # CrossScale-FPAR-Net：双向拼接对比学习架构
│   ├── 9_train_crossscale.py # 增强型训练脚本：支持中断恢复与快速验证
│   └── evaluate.py         # 评估脚本（同步支持单尺度与跨尺度模型推理）
├── data/
│   ├── s1_input/           # Sentinel-1 GRD 双极化 TIFF（VV+VH，10m，13景）
│   ├── s2_lable/           # Sentinel-2 FPAR 原始分辨率（20m，15景）
│   ├── s2_label_10m/       # S2 FPAR 重采样至 10m（已对齐 S1，13景）
│   └── output_fpar/        # 评估输出（预测对比图）
├── models/                 # 模型权重（.pth 文件）
│   ├── best_model.pth      # 最佳验证损失对应的完整 checkpoint
│   └── unet_fpar_epoch_*.pth  # 每 10 epoch 保存的权重
├── runs/                   # TensorBoard 日志目录 (U-Net)
├── runs_transformer/       # TensorBoard 日志目录 (Transformer)
├── run.bat                 # Windows 一键运行菜单（训练/评估/TensorBoard）
├── README.md               # 本文件
└── CHANGELOG.md            # 修改历史记录
```

## 环境要求

| 依赖 | 版本要求 |
|------|---------|
| Python | 3.10+ |
| PyTorch | 2.6+（CUDA 12.6，支持 RTX 50 系列） |
| rasterio | 最新稳定版 |
| numpy | 最新稳定版 |
| tqdm | 最新稳定版 |
| matplotlib | 最新稳定版 |
| tensorboard | 最新稳定版 |

### 安装步骤

```bash
# 1. 创建 conda 虚拟环境
conda create -n fpar_project python=3.10 -y

# 2. 激活环境
conda activate fpar_project

# 3. 安装 GPU 版 PyTorch（CUDA 12.6）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 4. 安装其他依赖
pip install rasterio numpy tqdm matplotlib tensorboard
```

## 运行顺序

1. **数据对齐**：
   - S2 对齐 S1：`python src/1_align_data.py`
   - MODIS 对齐 S1/S2：`python src/7_align_modis.py`
2. **训练模型**：
   - U-Net Baseline: `python src/4_train.py`
   - Dual-Stream Transformer: `python src/6_train_transformer.py`
   - CrossScale FPAR Fusion (V7): `python src/9_train_crossscale.py`
3. **评估模型**：`python src/evaluate.py`
   - 对所有有效 S1↔S2 配对做全图滑窗推理
   - 输出评估汇总表格，计算 R² 和 RMSE，生成含时间戳的 1×3 对比图

或直接运行 `run.bat` 通过带编号的新菜单进行选择即可。

## 模型架构

### 1. 标准 U-Net (`src/3_unet.py`)
- **基于**：Encoder-Decoder + Skip Connection
- **输入**：2 通道（S1 VV + VH）
- **输出**：1 通道（FPAR 预测值）
- **参数量**：约 31.04 M
- **损失函数**：MSELoss

### 2. Dual-Stream Transformer (`src/5_transformer_model.py`)
- **基于**：双路流提取 + 特征混合与还原映射 (Disaggregation)
- **Local CNN Stream**：4 层卷积提取物理极化细节与边缘。
- **Global Transformer Stream**：Patch Embedding + 4 层自注意力层结合提取全局上下文。
- **Decoder**：混合两路特征并逐级通过反卷积还原尺度。
- **损失函数**：MSE + 0.5 * L1（混合损失）。

## 数据说明

- **S1 数据**：13 景 Sentinel-1 GRD IW 模式影像，时间跨度 2025/04 - 2025/09
- **S2 数据**：Sentinel-2 反演 FPAR 产品，最近邻时间匹配配对（≤5 天）
- **空间分辨率**：统一 10m
- **Patch 大小**：256×256 像素
