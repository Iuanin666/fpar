# E:\FPAR_project\src\evaluate.py
# =============================================================================
# FPAR 融合模型评估与可视化脚本
# 功能：
#   1. 对所有（或指定）日期的 S1/S2 影像做全图滑窗推理
#   2. 使用最近邻时间匹配（与 2_dataset.py 一致），而非位置索引匹配
#   3. 计算有效像素范围内的 R²（决定系数）和 RMSE（均方根误差）
#   4. 绘制并保存 1×3 对比图（中文字体支持）
#   5. 输出所有场景的汇总评估表
# =============================================================================

import os
import sys
import re
import math
import argparse
from datetime import datetime

import numpy as np
import torch
import rasterio
import matplotlib
matplotlib.use("Agg")   # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

# ── 中文字体配置（解决 CJK 字符无法显示的问题）────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

# PyTorch 2.x 推荐使用 torch.amp，旧版兼容 torch.cuda.amp
try:
    from torch.amp import autocast as _autocast  # PyTorch >= 2.0
    def autocast(enabled=True): return _autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast  # PyTorch < 2.0

# ── 模块路径处理 ─────────────────────────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from importlib import import_module
unet_module    = import_module("3_unet")
tf_module      = import_module("5_transformer_model")
cs_module      = import_module("8_crossscale_model")
DualStreamUNet        = unet_module.DualStreamUNet
DualStreamTransformer = tf_module.DualStreamTransformer
CrossScaleFPARNet      = cs_module.CrossScaleFPARNet


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         用户配置区（按需修改）                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 路径配置 ──────────────────────────────────────────────────────────
S1_DIR         = r"E:\FPAR_project\data\s1_input\aligned_output"
LABEL_DIR      = r"E:\FPAR_project\data\s2_label_aligned"
DEM_PATH       = r"E:\FPAR_project\data\s1_input\S1_Terrain_10m.tif"
UNET_MODEL_PATH = r"E:\FPAR_project\models\unet_best_model.pth"
TRANSFORMER_MODEL_PATH = r"E:\FPAR_project\models\transformer_best_model.pth"
CROSSSCALE_MODEL_PATH = r"E:\FPAR_project\models\crossscale_best_model.pth"
OUTPUT_DIR     = r"E:\FPAR_project\data\output_fpar"

# ── 测试目标配置 ──────────────────────────────────────────────────────
# 设为 "all" 则评估所有可配对的场景（时间差 ≤ MAX_TIME_DIFF 天）
# 设为列表 如 ["20250719"] 则仅评估指定日期
TEST_DATES = "all"
MAX_TIME_DIFF = 180   # SenRVM策略: 放宽硬拼接时间差限制至 180 天，将时间差作为动态权重特征交还给注意力层去挖掘

# ── 模型结构参数（需与训练时一致） ───────────────────────────────────
IN_CHANNELS = 7    # VV + VH + 3-Terrain + Delta + DOY
PATCH_SIZE  = 256  # 推理时的滑窗大小（同训练 PATCH_SIZE）

# ── 推理配置 ──────────────────────────────────────────────────────────
STRIDE_RATIO = 0.5   # 滑窗步长 = PATCH_SIZE × STRIDE_RATIO（0.5 = 50% 重叠）
USE_AMP      = True  # 混合精度推理（CPU 环境自动禁用）

# ── 可视化配置 ────────────────────────────────────────────────────────
DISPLAY_DOWNSAMPLE = 4   # 降采样倍数（不影响指标计算）


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            辅助函数                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _parse_s1_date(filename: str) -> datetime:
    """从 S1 文件名提取日期（_YYYYMMDT... 格式）。"""
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法解析 S1 日期：{filename}")


def _parse_s2_date(filename: str) -> datetime:
    """从 S2 文件名提取日期（_YYYYMMDD_ 格式）。"""
    match = re.search(r'_(\d{8})_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法解析 S2 日期：{filename}")


def find_all_pairs(s1_dir: str, label_dir: str, max_time_diff: int = 180):
    """
    使用 S2 (FPAR 真值) 作为基准，匹配最近邻时间的 S1。
    返回 [(s1_path, s2_path, s1_date_str, s2_date_str, time_diff), ...]
    """
    s1_files = sorted([f for f in os.listdir(s1_dir)    if f.lower().endswith('.tif')])
    s2_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.tif')])

    # 解析所有 S1 日期
    s1_dates = []
    for s1f in s1_files:
        try:
            s1_dates.append(_parse_s1_date(s1f))
        except ValueError:
            s1_dates.append(None)

    pairs = []
    for j, s2f in enumerate(s2_files):
        try:
            d2 = _parse_s2_date(s2f)
        except ValueError:
            continue

        # 找最近的 S1
        best_idx, best_delta = -1, float("inf")
        for i, d1 in enumerate(s1_dates):
            if d1 is None:
                continue
            delta = abs((d2 - d1).days)
            if delta < best_delta:
                best_delta = delta
                best_idx = i

        if best_idx < 0 or best_delta > max_time_diff:
            continue

        pairs.append((
            os.path.join(s1_dir, s1_files[best_idx]),
            os.path.join(label_dir, s2f),
            s1_dates[best_idx].strftime("%Y%m%d"),
            d2.strftime("%Y%m%d"),
            best_delta,
        ))

    return pairs


def sliding_window_inference(
    model, s1_data: np.ndarray, dem_data: np.ndarray, 
    dem_min: list, dem_max: list,
    delta_norm: float, doy_norm: float,
    patch_size: int, stride: int,
    device: torch.device, use_amp: bool
) -> np.ndarray:
    """
    对整景 S1 影像做滑动窗口推理，返回与输入同空间分辨率的预测 FPAR 图（H, W）。
    输入包含 7 个通道逻辑：[VV, VH, Elevation, Slope, Aspect, DeltaT, DOY]
    """
    _, H, W = s1_data.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)

    model.eval()

    tops  = list(range(0, H - patch_size + 1, stride))
    lefts = list(range(0, W - patch_size + 1, stride))
    if tops[-1] + patch_size < H: tops.append(H - patch_size)
    if lefts[-1] + patch_size < W: lefts.append(W - patch_size)

    # 归一化常量 (需与 2_dataset.py 锁死一致)
    VV_MIN, VV_MAX = -30.0, 5.0
    VH_MIN, VH_MAX = -35.0, 5.0

    print(f"  [推理] 正在执行 7-通道滑动窗口推理...")

    with torch.no_grad():
        for top in tops:
            for left in lefts:
                # 1. 提取物理 Patch
                p_s1 = s1_data[:, top:top+patch_size, left:left+patch_size].copy()
                p_dem = dem_data[:, top:top+patch_size, left:left+patch_size].copy()
                
                # 2. 物理通道归一化
                p_s1[0] = (np.clip(p_s1[0], VV_MIN, VV_MAX) - VV_MIN) / (VV_MAX - VV_MIN)
                p_s1[1] = (np.clip(p_s1[1], VH_MIN, VH_MAX) - VH_MIN) / (VH_MAX - VH_MIN)
                
                # 3. 地形通道自适应归一化
                p_dem_norms = []
                for i in range(3):
                    denom = dem_max[i] - dem_min[i]
                    norm = (p_dem[i] - dem_min[i]) / (denom if denom > 1e-6 else 1.0)
                    p_dem_norms.append(torch.from_numpy(norm[np.newaxis]).float())
                
                # 4. 构造 7 通道 Tensor: [2 S1, 3 Terrain, 1 Delta, 1 DOY]
                t_s1    = torch.from_numpy(p_s1).float()
                t_dem   = torch.cat(p_dem_norms, dim=0)
                t_delta = torch.full((1, patch_size, patch_size), delta_norm, dtype=torch.float32)
                t_doy   = torch.full((1, patch_size, patch_size), doy_norm, dtype=torch.float32)
                
                x = torch.cat([t_s1, t_dem, t_delta, t_doy], dim=0).unsqueeze(0).to(device)

                with autocast(enabled=use_amp):
                    y = model(x)

                pred_patch = y[0, 0].cpu().float().numpy()
                pred_sum[top:top+patch_size, left:left+patch_size] += pred_patch
                count[top:top+patch_size, left:left+patch_size]    += 1.0

    count = np.where(count == 0, 1.0, count)
    pred_map = pred_sum / count
    return pred_map


def compute_metrics(pred: np.ndarray, label: np.ndarray):
    """
    计算预测图与标签之间的 R² 和 RMSE。
    过滤无效像素（label ≤ 0 或 > 1，NaN/Inf）。
    """
    valid_mask = (
        np.isfinite(label) &
        np.isfinite(pred)  &
        (label >= 0.05)    &
        (label <= 1.0)
    )
    n_valid = valid_mask.sum()
    if n_valid < 2:
        print("  [警告] 有效像素不足（< 2），无法计算指标。")
        return float("nan"), float("nan")

    p = pred[valid_mask]
    t = label[valid_mask]

    rmse = math.sqrt(np.mean((p - t) ** 2))

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return r2, rmse


def plot_comparison(
    label: np.ndarray,
    pred: np.ndarray,
    r2: float,
    rmse: float,
    save_path: str,
    s1_date: str,
    s2_date: str,
    downsample: int = 1,
):
    """绘制 1×2 对比图 (Label vs Pred) 并保存。"""
    ds = max(1, downsample)

    lbl = label[::ds, ::ds]
    prd = pred[::ds, ::ds]

    # [核心修复] 对可视化应用严格的空间掩膜
    mask = lbl >= 0.05
    lbl_masked = np.where(mask, lbl, np.nan)
    prd_masked = np.where(mask, prd, np.nan)

    # 颜色映射
    cmap_fpar = plt.get_cmap("RdYlGn").copy()
    cmap_fpar.set_bad(color='#1a1a2e')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    # 子图 1：S2 真实 FPAR
    im1 = axes[0].imshow(lbl_masked, cmap=cmap_fpar, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"S2 FPAR Ground Truth ({s2_date})", color="white", fontsize=12, pad=10)
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # 子图 2：模型预测 FPAR
    metrics_str = (f"R2 = {r2:.4f}" if not math.isnan(r2) else "R2 = N/A") + " | " + \
                  (f"RMSE = {rmse:.4f}" if not math.isnan(rmse) else "RMSE = N/A")
    im2 = axes[1].imshow(prd_masked, cmap=cmap_fpar, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Predicted FPAR\n{metrics_str}", color="white", fontsize=12, pad=10)
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    plt.suptitle(f"FPAR Estimation Comparison (S1:{s1_date} <-> S2:{s2_date})", 
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [可视化] 对比图已保存至: {os.path.basename(save_path)}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                              主函数                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def evaluate_single(
    s1_path, s2_path, dem_global_data, dem_min, dem_max, s1_date, s2_date, time_diff,
    model, device, use_amp, stride, output_dir, model_tag="unet"
):
    """对单个 S1↔S2 配对执行评估，返回 (r2, rmse, save_path)。"""
    
    d1 = datetime.strptime(s1_date, "%Y%m%d")
    d2 = datetime.strptime(s2_date, "%Y%m%d")
    
    # 归一化元数据特征 (需与 2_dataset.py 严格一致)
    delta_norm = (d1 - d2).days / 30.0
    doy_norm   = float(d1.timetuple().tm_yday) / 366.0

    print(f"\n{'─'*60}")
    print(f"  评估场次: S1={s1_date} <-> S2={s2_date} (DOY: {d1.timetuple().tm_yday})")
    print(f"  时间偏差: {delta_norm*30:.1f} 天 (Norm: {delta_norm:.4f})")
    print(f"{'─'*60}")

    # 读取影像
    with rasterio.open(s1_path) as src:
        s1_data = src.read().astype(np.float32)
    with rasterio.open(s2_path) as src:
        label_data = src.read(1).astype(np.float32)

    # 3-路尺寸硬对齐 (S1, S2, DEM)
    min_h = min(s1_data.shape[1], label_data.shape[0], dem_global_data.shape[1])
    min_w = min(s1_data.shape[2], label_data.shape[1], dem_global_data.shape[2])
    
    s1_data     = s1_data[:, :min_h, :min_w]
    label_data  = label_data[:min_h, :min_w]
    dem_aligned = dem_global_data[:, :min_h, :min_w]
    
    print(f"  对齐尺寸: {min_h} x {min_w}")
    s1_data = np.nan_to_num(s1_data, nan=0.0, posinf=0.0, neginf=0.0)
    label_data = np.nan_to_num(label_data, nan=0.0, posinf=0.0, neginf=0.0)
    label_data[label_data < 0.05] = 0.0

    # 执行滑窗推理
    pred_map = sliding_window_inference(
        model, s1_data, dem_aligned, dem_min, dem_max,
        delta_norm, doy_norm,
        PATCH_SIZE, stride, device, use_amp
    )
    pred_map = np.clip(pred_map, 0.0, 1.0)

    # 对齐尺寸
    min_h = min(pred_map.shape[0], label_data.shape[0])
    min_w = min(pred_map.shape[1], label_data.shape[1])
    pred_map   = pred_map[:min_h, :min_w]
    label_data = label_data[:min_h, :min_w]

    # 诊断 R2 < 0 的原因
    valid_mask = (label_data >= 0.05) & (label_data <= 1)
    if valid_mask.any():
        print(f"  [诊断] 预测值: mean={pred_map.mean():.4f}, std={pred_map.std():.4f}")
        print(f"  [诊断] 真值  : mean={label_data[valid_mask].mean():.4f}, std={label_data[valid_mask].std():.4f}")

    # 计算指标
    r2, rmse = compute_metrics(pred_map, label_data)
    print(f"  R2 = {r2:.6f}  |  RMSE = {rmse:.6f}")

    # 绘图（加入模型前缀）
    timestamp = datetime.now().strftime("%H%M%S")
    save_fname = f"{model_tag.upper()}_eval_S1_{s1_date}_S2_{s2_date}_{timestamp}.png"
    save_path  = os.path.join(output_dir, save_fname)
    plot_comparison(
        label=label_data, pred=pred_map,
        r2=r2, rmse=rmse, save_path=save_path,
        s1_date=s1_date, s2_date=s2_date,
        downsample=DISPLAY_DOWNSAMPLE,
    )

    return r2, rmse, save_path

    return r2, rmse, save_path


def main():
    parser = argparse.ArgumentParser(description="评估 FPAR 模型")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "transformer", "crossscale"],
                        help="选择要评估的模型架构")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  FPAR 模型评估脚本 ── 测试模型: {args.model.upper()}")
    print(f"{'═'*60}\n")

    # ── 0. 设备检测与 GPU 优化 ─────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # GPU 优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

        print(f"  [GPU] {gpu_name} ({vram_gb:.1f} GB)")
        print(f"  [GPU] cuDNN Benchmark: 开启 | TF32: 开启 | AMP: {'开启' if USE_AMP else '关闭'}")
    else:
        device = torch.device("cpu")
        print(f"  [设备] CPU（未检测到 CUDA GPU）")

    use_amp = USE_AMP and (device.type == "cuda")

    # ── 1. 查找所有有效配对 ─────────────────────────────────────────────
    print(f"\n[步骤 1] 查找 S1↔S2 配对（最近邻时间匹配，≤{MAX_TIME_DIFF} 天）...")
    all_pairs = find_all_pairs(S1_DIR, LABEL_DIR, MAX_TIME_DIFF)
    
    # ── 1.5 预加载 DEM ──
    print(f"  [数据] 正在预加载 DEM: {DEM_PATH} ...")
    with rasterio.open(DEM_PATH) as src:
        # 阶段 7: 读取前 3 个波段 (海拔, 坡度, 坡向)
        dem_data = src.read((1, 2, 3)).astype(np.float32)
        # 计算全局 Min-Max 用于自适应归一化
        dem_min = [np.nanmin(dem_data[i]) for i in range(3)]
        dem_max = [np.nanmax(dem_data[i]) for i in range(3)]
    print(f"  [数据] DEM 载入成功，Shape: {dem_data.shape}")
    for i, name in enumerate(["海拔(B1)", "坡度(B2)", "坡向(B3)"]):
        print(f"    - {name}: Min={dem_min[i]:.2f}, Max={dem_max[i]:.2f}")

    if not all_pairs:
        raise RuntimeError("未找到任何有效的 S1↔S2 配对！请检查数据目录。")

    # 根据 TEST_DATES 过滤
    if isinstance(TEST_DATES, list):
        pairs = [p for p in all_pairs if p[2] in TEST_DATES]
        if not pairs:
            raise RuntimeError(f"在有效配对中未找到指定日期 {TEST_DATES}")
    else:
        pairs = all_pairs  # "all"

    print(f"  找到 {len(all_pairs)} 个有效配对，将评估 {len(pairs)} 个场景：")
    for s1p, s2p, d1, d2, dt in pairs:
        print(f"    S1={d1}  <->  S2={d2}  (差 {dt} 天)")

    # ── 2. 加载模型 ───────────────────────────────────────────────────
    if args.model == "unet":
        model_path = UNET_MODEL_PATH
        print(f"\n[步骤 2] 加载 Dual-Stream U-Net 模型: {model_path}")
        model = DualStreamUNet(in_channels=IN_CHANNELS, out_channels=1).to(device)
    elif args.model == "transformer":
        model_path = TRANSFORMER_MODEL_PATH
        print(f"\n[步骤 2] 加载 Transformer 模型: {model_path}")
        model = DualStreamTransformer(in_channels=IN_CHANNELS, out_channels=1, img_size=PATCH_SIZE).to(device)
    else:
        model_path = CROSSSCALE_MODEL_PATH
        print(f"\n[步骤 2] 加载 CrossScale 模型: {model_path}")
        model = CrossScaleFPARNet(in_channels=IN_CHANNELS, patch_size=PATCH_SIZE).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型文件：{model_path}\n"
            f"请先完成相应模型的训练后再执行本脚本。"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint.get("epoch", "?")
        # 支持多种模型可能的 Loss Key
        val_loss = checkpoint.get("val_loss") or checkpoint.get("best_val_loss", "?")
        
        if isinstance(val_loss, (float, int)):
            loss_str = f"{val_loss:.6f}"
        else:
            loss_str = str(val_loss)
            
        print(f"  [OK] 权重加载成功（Epoch {epoch}，Val Loss={loss_str}）")
    else:
        model.load_state_dict(checkpoint)
        print("  [OK] 权重加载成功")

    # ── 3. 逐场景评估 ─────────────────────────────────────────────────
    print(f"\n[步骤 3] 开始逐场景评估...")
    stride = max(1, int(PATCH_SIZE * STRIDE_RATIO))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for s1_path, s2_path, s1_date, s2_date, time_diff in pairs:
        r2, rmse, save_path = evaluate_single(
            s1_path, s2_path, dem_data, dem_min, dem_max, s1_date, s2_date, time_diff,
            model, device, use_amp, stride, OUTPUT_DIR, model_tag=args.model
        )
        results.append((s1_date, s2_date, time_diff, r2, rmse, save_path))

    # ── 4. 汇总结果 ───────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  评估汇总（共 {len(results)} 个场景）")
    print(f"{'═'*72}")
    print(f"  {'序号':>4}  {'S1日期':^10}  {'S2日期':^10}  {'时差':^4}  {'R²':^10}  {'RMSE':^10}")
    print(f"{'─'*72}")

    r2_list, rmse_list = [], []
    for i, (d1, d2, dt, r2, rmse, sp) in enumerate(results):
        r2_str  = f"{r2:.4f}"  if not math.isnan(r2)   else "  N/A  "
        rmse_str = f"{rmse:.4f}" if not math.isnan(rmse) else "  N/A  "
        print(f"  {i+1:>4}  {d1:^10}  {d2:^10}  {dt:^4}  {r2_str:^10}  {rmse_str:^10}")
        if not math.isnan(r2):
            r2_list.append(r2)
        if not math.isnan(rmse):
            rmse_list.append(rmse)

    print(f"{'─'*72}")
    if r2_list:
        print(f"  平均  {'':^10}  {'':^10}  {'':^4}  {np.mean(r2_list):^10.4f}  {np.mean(rmse_list):^10.4f}")
    print(f"{'═'*72}")
    print(f"  所有对比图已保存至: {OUTPUT_DIR}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
