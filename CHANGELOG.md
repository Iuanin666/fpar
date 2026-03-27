## [2026-03-27 10:30] 🔴 V7.1 致命架构修复：知识蒸馏模式 + PHRU 损失激活

### 致命 Bug 修复
| Bug | 问题描述 | 修复方案 |
|-----|----------|----------|
| **训练-推理脱节** | Transformer 训练时直接吃真实 MODIS，推理时被迫用自生 PLRU，导致模型"偷懒"不学 S1 纹理，推理时输出均值图 (R²<0)。 | **强制统一**：Transformer 无论训练还是推理，都只使用自生成的 `plru`。MODIS 降级为知识蒸馏教师，仅通过 `L_cons` 提供监督。 |
| **PHRU 死代码** | `DisaggregationModule` 生成了 `phru`，但 Loss 函数从未使用它，该分支无梯度回传。 | **激活 `L_phru`**：新增 `L_phru = HuberLoss(phru, S2_label)`，权重 0.5，迫使 S1 特征必须包含空间引导信息。 |

### 修改文件
| 文件 | 修改 |
|------|------|
| `src/8_crossscale_model.py` | 1. `forward()`: 删除 `if modis_lr` 的 Transformer 分支，统一走 `plru`。<br>2. `CrossScaleLoss`: 新增 `L_phru` 分支 + 更新总损失公式。 |
| `src/9_train_crossscale.py` | 同步 `loss_details_sum` 加入 `L_phru`，Epoch 日志打印新增 `phru=` 字段。 |

---

## [2026-03-26 10:00] 🌐 V7 跨尺度时空融合：双向拼接对比学习与 MODIS 整合

### 核心革新
基于"双向拼接对比学习 (Bi-directional Patch Contrastive Learning)"架构，正式引入 MODIS 旬合成 FPAR 产品作为低分辨率 (LRU) 先验物理约束，实现 10m ↔ 500m 的跨尺度特征 disaggregation 与聚集映射。

### 修复与新特性
| 维度 | 修改内容 |
|------|----------|
| **数据处理** | 1. **MODIS 对齐** (`7_align_modis.py`): 1000m→500m 重采样 + 重投影裁剪。<br>2. **严格 0.05 过滤**: 抛弃 0 值逻辑，改用 `NaN` 彻底隔离低 FPAR 噪声区域。 |
| **模型架构** | **CrossScale-FPAR-Net** (`8_crossscale_model.py`): <br>- **Aggregation**: 高分特征聚合生成 Pseudo-LRU (PLRU)。<br>- **Disaggregation**: MODIS 引导特征拆解生成 Pseudo-HRU (PHRU)。<br>- **Triple Loss**: `L_cont` (纹理) + `L_cons` (物理一致性) + `L_temp` (时间连续)。 |
| **训练功能** | **增强型训练脚本** (`9_train_crossscale.py`):<br>- **中断恢复**: 支持 `--resume` 自动加载最优权重断点续训。<br>- **快速验证**: 支持 `--test_epochs` 小批量试跑，防止无效长跑。 |
| **数据结构** | 1. **S2 核心对齐 (Pivot logic)**: 重写 `CrossScaleDataset` 和 `evaluate.py`，改为以 S2 为基准寻找最近邻的 S1 和 MODIS，确保物理一致性线。 |
| **架构修正** | 1. **断绝“模型偷懒”故障**: 修改 `8_crossscale_model.py` 的 Transformer 输入，强制训练/推理统一使用生成的 `plru`，禁止直接读取 MODIS 真值特征（MODIS 仅作监督）。<br>2. **激活 PHRU 损失**: 新增 `L_phru` 到 `CrossScaleLoss`，强制 `f4` 特征包含足够的空间拆解指引。 |
| **Loss 调优 (V7.2)** | 1. **对抗 MODIS 平滑**: 调低 `lambda_cons` (0.3→0.1) 减弱 MODIS 低通滤波属性；调高 Pearson 散度权重 (0.5→1.5) 强制网络在 10m 级别拉高空间方差 ($std$)。<br>2. **训练周期延长**: `NUM_EPOCHS` 提升至 150，耐心值升至 35，适配大容量 Transformer 所需的拟合周期。 |
| **系统同步** | 1. **`run.bat`**: 菜单升级至 V7 版。<br>2. **验证与打印**: 修复 `evaluate.py` 实例化与格式化报错。训练脚本控制台实时显示 `cont/cons/phru` 三重损失。 |

---

## [2026-03-14 11:15] 🎯 V6 精度地形融合：3-波段自适应归一化

### 核心革新
在 V5 基础上进一步挖掘地形潜力，引入海拔、坡度、坡向全维度特征，并实现数据驱动的自适应归一化，解决人为假设范围导致的数据失真。

### 修复内容
| 文件 | 修改 |
|------|------|
| `src/2_dataset.py` | 1. **全地形特征**: 同时加载海拔 (B1)、坡度 (B2)、坡向 (B3)。<br>2. **自适应归一化**: 自动计算地形影像的全局 Min-Max，将地形各波段精确缩放至 $[0, 1]$，大幅提升跨区域稳定性。<br>3. **7-通道输出**: 最终输入扩充为: `[VV, VH, Elevation, Slope, Aspect, DeltaT, DOY]`。 |
| `src/3_unet.py` | `DualStreamUNet` 空间分支接收通道从 3 提升至 5，适配 S1 + 全地形特征。 |
| `src/5_transformer_model.py` | `DualStreamTransformer` 默认输入通道数对齐至 7。 |
| `src/4_train.py` & `src/6_train_transformer.py` | 同步更新 `IN_CHANNELS = 7`。 |
| `src/evaluate.py` | 重构推理逻辑：同步 7-通道架构，在推理前实时计算地形影像的 Min-Max，确保推理与训练环境锁死一致。 |

---

## [2026-03-14 10:30] ⛰️ V5 物理先验增强：引入 DEM 地形与 DOY 季节物候特征 (初步)

### 核心改进
为了打破 S1 雷达信号与 S2 光学 FPAR 之间“二对一”的病态映射困局，我们尝试引入物理先验（地形）和时间先验（季节物候）。

### 修复内容
| 文件 | 修改 |
|------|------|
| `src/2_dataset.py` | 1. **三路对齐**：实现 S1, S2, DEM 的严格像素级对齐裁剪。<br>2. **季节先验**：计算归一化 `DOY` 作为第 5 通道。<br>3. **地形初探**：加载 `S1_Terrain_10m.tif` 第 1 波段。 |
| `src/3_unet.py` | 架构升级为 5 通道输入 Dual-Stream 模式。 |
| `src/evaluate.py` | 推理管线同步适配 5 通道。 |

---

## [2026-03-13 19:10] 🏎️ V4 性能压榨：引入常驻内存级 IO 缓存及 DataLoader 终极提速

### 核心痛点
此前的 Dataloader 针对每一张返回的图像切片 (Patch) 都使用了 `rasterio.open` 去读取整个 700MB 大小的 S1 和 S2 原生 TIFF 图。由于每个 Epoch 在同一场景内会执行数百次读取，引发了极其严重的磁盘 IO 瓶颈，导致 Ryzen 9 和 RTX 5060 基本处于“断水断粮”的空置状态，进而形成了心电图搬的**脉冲状利用率**。

### 修复内容
| 文件 | 修改 |
|------|------|
| `src/2_dataset.py` | 1. 引入了内存常驻缓冲区 `self.cache`。<br>2. 仅在首次遇到某一场景时才从磁盘进行完整的 `rasterio.read` 并存入字典；后续所有切片读取全部绕过多级总线，变为百纳秒级 RAM 极速响应！彻底拔除磁盘 IO 瓶颈。 |
| `src/4_train.py` | U-Net: 从 8 回退至安全值 `BATCH_SIZE = 16`，强制关闭多进程 `NUM_WORKERS = 0` 防止 Windows spawn 引发内存炸雷现象。 |
| `src/6_train_transformer.py` | Transformer: 受限于 VRAM 上限退回 `BATCH_SIZE = 8`，关闭多进程。 |

---

## [2026-03-13 19:15] 🚑 紧急修复：Windows 多进程 DataLoader 导致内存炸雷 (OOM) 及卡死问题

### 问题根因
刚才的 V4 提速版本中我们成功把数据塞入了内存中，但忽视了一个致命机制：**在 Windows 操作系统下，PyTorch DataLoader 使用 `spawn` 方式启动多进程，而不是 Linux 的 `fork`（写时复制机制）！**
这意味着当设置 `NUM_WORKERS = 8` 时，系统试图把那 13GB 大小的内存级 Dataset 对象**通过序列化完整地复制 8 份**分发给 8 个独立子进程。这瞬间击穿了 32GB 内存条的上限（13GB × 8 = 104GB），导致直接系统卡死瘫痪。并且因为 3 通道的引入，把 Batch Size 拉大到 24 也超出了 RTX 5060 8G 的物理上限引发 GPU OOM。

### 修复方案
非常暴力有效：既然缓存已经在内存里了，抽取切片的速度在主线程内也是达到纳秒级的！**因此我们直接禁用多进程！** 
1. `NUM_WORKERS` 强制设为 `0`：只有主线程独占一份高达 10~15GB 的平稳缓存数据。
2. U-Net 显存压榨退回安全上限：`BATCH_SIZE = 16`。
3. Transformer 参数量大：`BATCH_SIZE = 8`。
这样就实现了：“既不狂读硬盘，又绝不炸内存”的终极单进程极速读取方案！

---

## [2026-03-13 18:55] 🚀 V3 架构突破：SenRVM 时序元数据融合与 0.05 物理掩膜剔除

### 核心革新
基于顶刊 SenRVM 论文的思路，彻底解决了 S1↔S2 因时间差 5 天约束导致“训练样本枯竭”以及“大量地表脏数据导致梯度爆炸”的两大核心痛点。

### 修复内容
| 文件 | 修改 |
|------|------|
| `src/2_dataset.py` | 1. **放宽硬对齐限制**：移除 `max_time_diff = 5`，强行引入距对应 S1 最近的无云 S2 标签（最大放宽至 180 天防跨季）。<br>2. **融合元数据层**：计算 `(S1_date - S2_date) / 30.0` 构建为网络输入的第三通道 `delta_channel`。<br>3. **强制截断无效背景**：提取 Patch 时截去 FPAR `< 0.05` 的有毒区域。 |
| `src/4_train.py` | 模型输入通道 `IN_CHANNELS` 更换为 3。`MaskedMSELoss` 触发阈值从 `0.01` 收紧为严格的 `0.05`，进一步清除因为阴影引起的假零值。 |
| `src/6_train_transformer.py` | 同上，更换通道配置及混合损失的截断底线。 |
| `src/evaluate.py` | 推理机制升级：动态生成 `delta_channel` 以适配预测滑动窗口输入。RMSE、R²和绘图阶段底层过滤一视同仁全部对齐至 `0.05`，只评估严谨有效植被。 |

---

## [2026-03-13 18:45] 🚨 紧急修复：掩膜损失生效失败与梯度消失问题

### 修复内容
| 文件 | 修改 |
|------|------|
| `src/4_train.py` | 修复上个版本因代码匹配失败导致 `MaskedMSELoss` 未实际应用进训练循环的问题。现在模型已确实过滤了所有水体/无效背景。增加了**梯度裁剪 (clip_grad_norm_)** 防止 Mask 过滤带来的方差波动 |
| `src/3_unet.py` | 对 U-Net 所有积层应用 **Kaiming Normal (He) 权重初始化**，以摆脱默认小权重导致的梯度消失问题。对最后的输出层添加 `bias = -0.4` （对应 Sigmoid 均值约 0.4），让模型起步时预测期望直接对齐实际植被均值，不再掉入全局网络崩溃输出 0.17 的陷阱 |

---

## [2026-03-12 03:30] ⚠️ 修复致命偏差：引入空间掩膜损失 (Masked Loss)

### 问题源头
从可视化的真值中发现，S2 数据中存在大量由于水体/城镇/空缺导致的无效背景（FPAR = 0）。而在训练时，模型使用全局 `MSELoss`，为了迎合这张图里巨量的背景区 0 值，模型拼命向下输出，导致真正植被区的预测均值被严重拉低，从而得到 R² 为负的糟糕结果。

### 修复内容
| 文件 | 修改 |
|------|------|
| `src/4_train.py` | 引入 `MaskedMSELoss`，在训练计算 Loss 时完全忽略 `label <= 0.01` 的像素点及其梯度，迫使模型**只在有效面学习** |
| `src/6_train_transformer.py` | 引入支持 Mask 过滤像素的 `MaskedMixedLoss`，功能同上 |
| `src/evaluate.py` | 1. `compute_metrics()` 中，**R² 和 RMSE 的计算严格屏蔽了 `label <= 0.01` 的区域**，确保只评估真实植被表现<br>2. `plot_comparison()` 渲染出图时，将预测图与真值图同时将背景区域屏蔽为深色底色透明，解决出图干扰问题 |

---

## [2026-03-12 03:00] 🔴 V2 数据管线重大修复（解决 R²<0 根因）

### 诊断结果
通过实测 S1 数据发现 **5 个致命问题**：
1. S1 VV 实际范围 [-42, +14] dB，VH [-34, +24] dB。原代码 `/30` 归一化假设 [-30,0] 完全错误
2. 62.6% S2 FPAR 像素为 0（水体/裸地/nodata），且存在负值 (min=-0.47)
3. 仅 120 个训练样本（6景×20 patches）训练 31M 参数，参数/样本比 258000:1
4. 无任何数据增强
5. 评估时无效像素（label=0）参与 R² 计算

### 修改文件
| 文件 | 修改 |
|------|------|
| `src/2_dataset.py` | 1. 增加 S1 和 S2 读取时的 最小尺寸对齐，修复两边差 1 个像素导致的裁剪越界引起 DataLoader 崩溃 <br> 2. `/30` → per-channel min-max clamp [0,1]；无效 patch 过滤（有效 FPAR<30% 跳过）；数据增强（翻转+旋转）；samples 20→200 |
| `src/evaluate.py` | 同步归一化；评估过滤阈值 `>0` → `>0.01` |
| `src/4_train.py` | samples 20→200，验证 10→50 |
| `src/6_train_transformer.py` | 同上 |

## [2026-03-12 02:51] 修复 `evaluate.py` 尺寸对齐 Bug
- 修复了因为增加 argparse 导致代码合并时丢失了“对齐S1和S2尺寸差异 (8421 vs 8420)”的裁剪代码段问题。

### 继续修复 evaluate.py
- 在输出的图像文件名中加上了 `HHMMSS` 时间戳（如 `eval_S1_20250426_S2_20250427_220510.png`），解决了每次评估可能互相覆盖的问题
- 在计算指标前加入了 `pred` 和 `label` 统计信息（mean, std, min, max）的打印。
- **关于 R² 为负数：** 并不是代码计算错误。当模型预测出的方差极小（预测近乎常数/均值图像）时，残差平方和（MSE）可能会大于真实标签数据的固有方差，此时 `1 - MSE/Var(y)` 就会得到负数。这通常是因为模型尚未充分学习（过拟合在很小的数据分布上）导致的。

### 创建双流 Transformer 模型
根据架构图新增了模型及其训练脚本（暂不引入低分率 MODIS 数据，关注核心特征混合）。
- **`src/5_transformer_model.py`**: 构建了 `DualStreamTransformer`。包含两条主路：
  - CNN Stream（局部特征流）：提取物理边缘与细节
  - Transformer Stream（全局特征流）：用 Patch Embedding 结合自注意力学习全局空间映射与混合权重
  - Decoder 融合了局部和全局特征流进行逐级缩放解码（Disaggregation）。
- **`src/6_train_transformer.py`**: 继承了已沉淀的良好训练配置（`NUM_WORKERS=4`, `BATCH_SIZE=8`, cuDNN benchmark, TF32），避免重复踩坑。另外使用了混合 Loss（MSE + 0.5 * L1）帮助学习高频细节。
- **`run.bat`**: 已将 1~4 的菜单扩充为 1~6，新增了 `Run Transformer Training` 和专属 TensorBoard 选项。

## [2026-03-11 21:44] 评估脚本重构

### 遇到的问题
1. **中文乱码**：matplotlib 默认 DejaVu Sans 不含 CJK 字符，图中中文全部显示为方块
2. **只测一个日期**：`TEST_DATE="20250719"` 硬编码，每次只评估一个场景
3. **配对错误**：`find_test_pair` 使用位置索引匹配 S1↔S2，而非最近邻时间匹配（与训练不一致）
4. **GPU 未优化**：评估推理未启用 cuDNN benchmark / TF32
5. **R² 为负数**：说明模型预测不如简单取均值，可能与训练数据不足或配对方式有关

### 修改内容（`src/evaluate.py`）
| 修改 | 说明 |
|------|------|
| 中文字体 | 添加 `plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']` |
| 多日期评估 | `TEST_DATES = "all"` 评估所有有效配对场景，不再固定单日期 |
| 最近邻匹配 | 新增 `find_all_pairs()` 使用与 `2_dataset.py` 相同的最近邻时间匹配算法 |
| GPU 优化 | 添加 cuDNN benchmark + TF32 + detailed GPU info |
| 汇总表格 | 评估完成后输出所有场景的 R²/RMSE 汇总表及平均值 |
| 图表标题 | 改为英文标题避免字体兼容问题 |

## [2026-03-11 19:28] GPU 性能优化（RTX 5060 Laptop 8GB）

### 遇到的问题
- `NUM_WORKERS=0` 导致数据加载在主线程执行，GPU 在等待数据时处于空闲
- `BATCH_SIZE=4` 未充分利用 8GB 显存，GPU 利用率低
- 未启用 cuDNN benchmark 和 TF32 加速

### 修改内容（`src/4_train.py`）
| 优化项 | 原值 | 新值 | 效果 |
|--------|------|------|------|
| BATCH_SIZE | 4 | 8 | GPU 计算密度翻倍，AMP fp16 下 VRAM 约用 2-3GB |
| NUM_WORKERS | 0 | 4 | 数据加载与 GPU 并行，消除 I/O 瓶颈 |
| persistent_workers | 无 | True | 避免每 Epoch 重建 worker |
| prefetch_factor | 无 | 2 | 每 worker 预缓冲 2 批数据 |
| cuDNN benchmark | 关 | 开 | 自动选择最优卷积算法 |
| TF32 | 关 | 开 | Tensor Core 加速矩阵运算 |

## [2026-03-11 19:18] 训练脚本 Bug 修复与验证集改进

### 遇到的问题
1. **`ReduceLROnPlateau` 报错**：PyTorch 2.10 移除了 `verbose` 参数，导致 `TypeError`
2. **验证集过小**：原逻辑每景仅取右下角 20% 区域的 1 个中心 Patch，6 景仅 6 个 Patch，无法充分评估模型
3. **训练顺序疑虑**：担心同一景的多个 Patch 连续出现导致景切换时 loss 突变

### 修改内容
| 文件 | 修改 |
|------|------|
| `src/4_train.py` | 移除 `verbose=True`；val `samples_per_image` 从 1 改为 10 |
| `src/2_dataset.py` | `__len__` 统一使用 `samples_per_image`；验证区从固定中心改为确定性随机采样（`random.Random(idx)` 种子保证跨 Epoch 可复现） |

### 效果
- 验证集：6 Patch → 60 Patch（6景×10个/景），覆盖验证区域更全面
- 训练顺序：DataLoader `shuffle=True` 已保证不同景的 Patch 在 Batch 中充分混合，无需担心

---

## [2026-03-11] 项目迁移与环境配置

### 背景
项目从旧电脑（用户 `lenovo`，路径 `F:\FPAR_project`，Anaconda 在 `C:\Users\lenovo\anaconda3`）拷贝至新电脑（用户 `iuanin`，路径 `E:\FPAR_project`，Conda 在 `E:\anaconda`）。

### 遇到的问题
1. **路径不一致**：所有源代码和配置文件中硬编码了旧电脑的绝对路径（`F:\FPAR_project`、`C:\Users\lenovo\anaconda3`），导致无法直接运行
2. **Python 环境缺失**：新电脑没有配置好的 Python 虚拟环境和依赖包
3. **GPU 兼容性**：新电脑 GPU 为 RTX 5060 Laptop（Blackwell 架构），需要 CUDA 12.6+ 和 PyTorch 2.6+ 才能支持

### 修改内容

#### 路径修正
| 文件 | 旧路径 | 新路径 |
|------|-------|-------|
| `src/1_align_data.py` | `F:\FPAR_project\data\...` | `E:\FPAR_project\data\...` |
| `src/4_train.py` | `F:\FPAR_project\...` | `E:\FPAR_project\...` |
| `src/evaluate.py` | `F:\FPAR_project\...` | `E:\FPAR_project\...` |
| `run.bat` | `C:\Users\lenovo\anaconda3\...` + `F:\FPAR_project\...` | conda 环境路径 + `E:\FPAR_project\...` |
| `.idea/misc.xml` | `C:\Users\lenovo\anaconda3` | conda 环境路径 |
| `.idea/FPAR_project.iml` | `C:\Users\lenovo\anaconda3` | conda 环境路径 |

#### 环境配置
- 创建 conda 虚拟环境 `fpar_project`（Python 3.10）
- 安装 PyTorch 2.6+（CUDA 12.6 版本）以支持 RTX 5060
- 安装 rasterio、numpy、tqdm、matplotlib、tensorboard

#### 文档新增
- 新增 `README.md` 项目说明文档
- 新增 `CHANGELOG.md` 修改历史记录（本文件）

---
