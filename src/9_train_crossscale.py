# E:\FPAR_project\src\9_train_crossscale.py
# =============================================================================
# CrossScale-FPAR-Net 训练脚本
# 核心功能：
#   1. 跨尺度数据加载 (S1 10m + S2 10m + MODIS 500m)
#   2. 三重损失函数训练 (L_cont + L_cons + L_temp)
#   3. --resume     : 自动从最近 checkpoint 恢复训练
#   4. --test_epochs: 仅运行 N 个 epoch 后停止（快速验证）
#   5. 每 5 epoch 保存 checkpoint（用于中断恢复）
#   6. 严格 0.05 NaN 过滤
# =============================================================================

import os
import sys
import re
import time
import argparse
import random
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import rasterio

try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = None
from tqdm import tqdm

# ── 模块导入 ──────────────────────────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from importlib import import_module
cs_module = import_module("8_crossscale_model")
CrossScaleFPARNet = cs_module.CrossScaleFPARNet
CrossScaleLoss    = cs_module.CrossScaleLoss


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         超参数与路径配置                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

S1_DIR         = r"E:\FPAR_project\data\s1_input\aligned_output"
LABEL_DIR      = r"E:\FPAR_project\data\s2_label_aligned"
DEM_PATH       = r"E:\FPAR_project\data\s1_input\S1_Terrain_10m.tif"
MODIS_DIR      = r"E:\FPAR_project\data\modis_aligned"
MODEL_SAVE_DIR = r"E:\FPAR_project\models"
LOG_DIR        = r"E:\FPAR_project\runs_crossscale"

HOLDOUT_DATE      = "20250719"
IN_CHANNELS       = 8        # V7.3: VV, VH, Elev, Slope, Aspect, DeltaT, DOY_sin, DOY_cos
BATCH_SIZE        = 4       # CrossScale 模型较大，从安全起步
PATCH_SIZE        = 256
LR_SIZE           = 5       # MODIS patch 在模型中的逻辑尺寸
SAMPLES_PER_IMAGE = 200
NUM_EPOCHS        = 150
LEARNING_RATE     = 2e-4     # 稍微激进，配合 Warmup 和大 Pearson 权重
WARMUP_EPOCHS     = 10       # 前 10 轮线性预热，避免初期梯度爆炸
EARLY_STOP_PATIENCE = 35
CHECKPOINT_EVERY   = 5     # 每 N epoch 保存一次 checkpoint
NUM_WORKERS        = 0
USE_AMP            = True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      多尺度数据集 (CrossScale)                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _parse_s1_date(filename):
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法解析 S1 日期: {filename}")

def _parse_s2_date(filename):
    match = re.search(r'_(\d{8})_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法解析 S2 日期: {filename}")

def _parse_modis_date(filename):
    """解析 MODIS 文件名: MODIS_FPAR_YYYYMMDD_500m.tif"""
    match = re.search(r'_(\d{8})_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法解析 MODIS 日期: {filename}")


class CrossScaleDataset(Dataset):
    """
    跨尺度 FPAR 数据集。
    返回: (s1_input, label_hr, modis_lr)
      - s1_input: (7, ps, ps) - 归一化后的 S1+Terrain+Meta
      - label_hr: (1, ps, ps) - S2 FPAR (无效区域为 NaN)
      - modis_lr: (1, lr_h, lr_w) - MODIS FPAR patch
    """

    def __init__(self, s1_dir, label_dir, modis_dir, dem_path,
                 patch_size=256, split="train", samples_per_image=200,
                 hold_out_date=None, min_valid_ratio=0.3, verbose=True):
        super().__init__()
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.modis_dir = modis_dir
        self.patch_size = patch_size
        self.split = split
        self.samples_per_image = samples_per_image
        self.min_valid_ratio = min_valid_ratio

        # ── 加载 DEM ───────────────────────────────────────────────────
        self.dem_data = None
        self.dem_min, self.dem_max = [], []
        if dem_path and os.path.exists(dem_path):
            with rasterio.open(dem_path) as src:
                self.dem_data = src.read((1, 2, 3)).astype(np.float32)
                for i in range(3):
                    self.dem_min.append(np.nanmin(self.dem_data[i]))
                    self.dem_max.append(np.nanmax(self.dem_data[i]))
            if verbose:
                print(f"[DEM] 载入成功: {self.dem_data.shape}")

        # ── 加载 MODIS ─────────────────────────────────────────────────
        modis_files = sorted([f for f in os.listdir(modis_dir) if f.endswith('.tif')])
        self.modis_dates = []
        self.modis_cache = {}
        for mf in modis_files:
            try:
                md = _parse_modis_date(mf)
                self.modis_dates.append((md, mf))
            except ValueError:
                continue

        # 预加载所有 MODIS 到内存
        for md, mf in self.modis_dates:
            with rasterio.open(os.path.join(modis_dir, mf)) as src:
                self.modis_cache[mf] = src.read(1).astype(np.float32)
        if verbose:
            print(f"[MODIS] 预加载 {len(self.modis_cache)} 个旬合成影像")

        # ── S1/S2 配对 ─────────────────────────────────────────────────
        s1_files = sorted([f for f in os.listdir(s1_dir) if f.lower().endswith('.tif')])
        s2_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.tif')])

        s2_dates = []
        for f in s2_files:
            try:
                s2_dates.append(_parse_s2_date(f))
            except ValueError:
                s2_dates.append(None)

        raw_pairs = []
        for j, s2f in enumerate(s2_files):
            d2 = s2_dates[j]
            if d2 is None:
                continue
            
            # --- 找最近的 S1 ---
            best_s1_idx, best_s1_delta = -1, float("inf")
            for i, s1f in enumerate(s1_files):
                try:
                    d1 = _parse_s1_date(s1f)
                    delta = abs((d2 - d1).days)
                    if delta < best_s1_delta:
                        best_s1_delta = delta
                        best_s1_idx = i
                except ValueError:
                    continue

            if best_s1_idx < 0 or best_s1_delta > 180:
                continue
            
            d1_best = _parse_s1_date(s1_files[best_s1_idx])

            # --- 找最近的 MODIS (以 S2 为主心骨) ---
            best_modis_f, best_modis_delta = None, float("inf")
            for md, mf in self.modis_dates:
                # 使用 S2 时间去找 MODIS
                md_delta = abs((d2 - md).days)
                if md_delta < best_modis_delta:
                    best_modis_delta = md_delta
                    best_modis_f = mf

            raw_pairs.append((
                s1_files[best_s1_idx], s2f, d1_best, d2,
                best_s1_delta, best_modis_f, best_modis_delta
            ))

        # 过滤 holdout
        self.pairs = []
        for p in raw_pairs:
            if hold_out_date and hold_out_date in p[0]:
                continue
            self.pairs.append(p)

        if verbose:
            print(f"\n[数据集 split='{split}'] 有效配对: {len(self.pairs)} 景")
            for i, (s1f, s2f, d1, d2, dt, mf, mdt) in enumerate(self.pairs):
                print(f"  {i+1}: S2={d2.strftime('%Y-%m-%d')} (基准) <-> S1={d1.strftime('%Y-%m-%d')} "
                      f"(差{dt}天) | MODIS={mf} (差{mdt}天)")

        if not self.pairs:
            raise RuntimeError("有效配对为 0!")

        self.cache = {}

    def __len__(self):
        return len(self.pairs) * self.samples_per_image

    def __getitem__(self, idx):
        scene_idx = idx // self.samples_per_image
        s1_fname, s2_fname, d1, d2, _, modis_fname, _ = self.pairs[scene_idx]

        # ── 缓存数据 ──────────────────────────────────────────────────
        if scene_idx not in self.cache:
            with rasterio.open(os.path.join(self.s1_dir, s1_fname)) as src:
                s1_data = src.read().astype(np.float32)
            with rasterio.open(os.path.join(self.label_dir, s2_fname)) as src:
                label = src.read(1).astype(np.float32)

            h_list = [s1_data.shape[1], label.shape[0]]
            w_list = [s1_data.shape[2], label.shape[1]]
            if self.dem_data is not None:
                h_list.append(self.dem_data.shape[1])
                w_list.append(self.dem_data.shape[2])
            min_h, min_w = min(h_list), min(w_list)

            s1_data = s1_data[:, :min_h, :min_w]
            label = label[:min_h, :min_w]
            dem = self.dem_data[:, :min_h, :min_w] if self.dem_data is not None else np.zeros((3, min_h, min_w), dtype=np.float32)

            self.cache[scene_idx] = (s1_data, label, dem)
        else:
            s1_data, label, dem = self.cache[scene_idx]

        _, H, W = s1_data.shape
        ps = self.patch_size

        # ── 空间分块 ──────────────────────────────────────────────────
        block_rows, block_cols = 5, 5
        block_h, block_w = H // block_rows, W // block_cols

        if self.split == "train":
            block_idx = random.randint(0, 19)
        else:
            rng = random.Random(idx)
            block_idx = rng.randint(20, 24)

        b_r, b_c = block_idx // block_cols, block_idx % block_cols
        b_top, b_left = b_r * block_h, b_c * block_w
        h_max = max(b_top, b_top + block_h - ps)
        w_max = max(b_left, b_left + block_w - ps)

        if self.split == "train":
            top = random.randint(b_top, h_max)
            left = random.randint(b_left, w_max)
        else:
            top = rng.randint(b_top, h_max)
            left = rng.randint(b_left, w_max)

        top = min(top, max(0, H - ps))
        left = min(left, max(0, W - ps))

        s1_patch = s1_data[:, top:top+ps, left:left+ps]
        label_patch = label[top:top+ps, left:left+ps]
        dem_patch = dem[:, top:top+ps, left:left+ps]

        # ── 无效 Patch 重采样 ─────────────────────────────────────────
        valid_ratio = np.sum(label_patch >= 0.05) / label_patch.size
        if self.split == "train" and valid_ratio < self.min_valid_ratio:
            for _ in range(5):
                t2 = random.randint(b_top, h_max)
                l2 = random.randint(b_left, w_max)
                t2 = min(t2, max(0, H - ps))
                l2 = min(l2, max(0, W - ps))
                lp = label[t2:t2+ps, l2:l2+ps]
                if np.sum(lp >= 0.05) / lp.size >= self.min_valid_ratio:
                    top, left = t2, l2
                    s1_patch = s1_data[:, top:top+ps, left:left+ps]
                    label_patch = lp
                    dem_patch = dem[:, top:top+ps, left:left+ps]
                    break

        # ── Tensor 转换 + 归一化 ──────────────────────────────────────
        s1_t = torch.from_numpy(s1_patch.copy()).float()
        s1_t = torch.nan_to_num(s1_t, nan=0.0, posinf=0.0, neginf=0.0)

        # S1 归一化
        VV_MIN, VV_MAX = -30.0, 5.0
        VH_MIN, VH_MAX = -35.0, 5.0
        s1_t[0] = (s1_t[0].clamp(VV_MIN, VV_MAX) - VV_MIN) / (VV_MAX - VV_MIN)
        s1_t[1] = (s1_t[1].clamp(VH_MIN, VH_MAX) - VH_MIN) / (VH_MAX - VH_MIN)

        # DEM 自适应归一化
        dem_norms = []
        for i in range(3):
            p = dem_patch[i]
            denom = self.dem_max[i] - self.dem_min[i]
            p_n = (p - self.dem_min[i]) / (denom if denom > 1e-6 else 1.0)
            dem_norms.append(torch.from_numpy(p_n[np.newaxis].copy()).float())

        # 元数据通道
        delta_days = abs((d1 - d2).days)  # 保留绝对天数，给 Loss 做时间衰减
        delta_norm = (d1 - d2).days / 30.0
        doy = float(d1.timetuple().tm_yday)
        doy_sin = math.sin(2 * math.pi * doy / 365.25)
        doy_cos = math.cos(2 * math.pi * doy / 365.25)
        delta_ch = torch.full((1, ps, ps), delta_norm, dtype=torch.float32)
        doy_sin_ch = torch.full((1, ps, ps), doy_sin, dtype=torch.float32)
        doy_cos_ch = torch.full((1, ps, ps), doy_cos, dtype=torch.float32)

        input_tensor = torch.cat([s1_t, *dem_norms, delta_ch, doy_sin_ch, doy_cos_ch], dim=0)  # (8, ps, ps)

        # ── 严格 0.05 过滤: 设为 NaN 而非 0 ──────────────────────────
        label_t = torch.from_numpy(label_patch.copy()).float().unsqueeze(0)
        label_t = torch.nan_to_num(label_t, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
        label_t[(label_t < 0.05) | (label_t > 1.0)] = float('nan')

        # ── MODIS 裁剪 ───────────────────────────────────────────────
        modis_full = self.modis_cache.get(modis_fname)
        if modis_full is not None:
            mH, mW = modis_full.shape
            # 计算 MODIS 与 S1 的像素比例
            scale_h = mH / H
            scale_w = mW / W
            m_top = int(top * scale_h)
            m_left = int(left * scale_w)
            m_h = max(1, int(ps * scale_h))
            m_w = max(1, int(ps * scale_w))
            m_top = min(m_top, max(0, mH - m_h))
            m_left = min(m_left, max(0, mW - m_w))
            modis_patch = modis_full[m_top:m_top+m_h, m_left:m_left+m_w]
            modis_t = torch.from_numpy(modis_patch.copy()).float().unsqueeze(0)  # (1, m_h, m_w)
        else:
            modis_t = torch.zeros((1, LR_SIZE, LR_SIZE), dtype=torch.float32)

        # ── 数据增强（仅训练集）────────────────────────────────────────
        if self.split == "train":
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [2])
                label_t = torch.flip(label_t, [2])
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [1])
                label_t = torch.flip(label_t, [1])
            k = random.randint(0, 3)
            if k > 0:
                input_tensor = torch.rot90(input_tensor, k, [1, 2])
                label_t = torch.rot90(label_t, k, [1, 2])

        # ── 标签中值滤波（去椒盐噪声，保留边缘）────────────────────
        if self.split == "train":
            valid_mask = ~torch.isnan(label_t)
            label_for_filt = torch.nan_to_num(label_t, nan=0.0)
            padded = torch.nn.functional.pad(label_for_filt, (1, 1, 1, 1), mode='reflect')
            patches = padded.unfold(1, 3, 1).unfold(2, 3, 1)  # (1, H, W, 3, 3)
            median_val = patches.contiguous().view(1, ps, ps, 9).median(dim=-1).values
            label_t = torch.where(valid_mask, median_val, label_t)

        return input_tensor, label_t, modis_t, delta_days


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          训练辅助函数                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def custom_collate(batch):
    """自定义 collate: MODIS patch 大小可能不同, 统一 resize 到 LR_SIZE."""
    inputs, labels, modis_list, delta_days_list = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    delta_days = torch.tensor(delta_days_list, dtype=torch.float32)  # (B,)
    # Resize MODIS patches to uniform size
    modis_resized = []
    for m in modis_list:
        m_r = torch.nn.functional.interpolate(
            m.unsqueeze(0), size=(LR_SIZE, LR_SIZE), mode='bilinear', align_corners=False
        ).squeeze(0)
        modis_resized.append(m_r)
    modis = torch.stack(modis_resized)
    return inputs, labels, modis, delta_days


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch, num_epochs, use_amp):
    model.train()
    epoch_loss = 0.0
    loss_details_sum = {'L_cont': 0, 'L_cons': 0, 'L_phru': 0, 'L_temp': 0}
    lr = get_lr(optimizer)

    pbar = tqdm(loader, desc=f"Train [{epoch+1:>3}/{num_epochs}]", ncols=110, leave=False)

    for inputs, labels, modis, delta_days in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        modis  = modis.to(device, non_blocking=True)
        delta_days = delta_days.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (autocast(_AMP_DEVICE, enabled=use_amp)
                   if _AMP_DEVICE else autocast(enabled=use_amp))
        with amp_ctx:
            pred_hr, plru, phru = model(inputs, modis)
            loss, details = loss_fn(pred_hr, plru, phru, labels, modis, delta_t=delta_days)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        for k in loss_details_sum:
            loss_details_sum[k] += details[k]
        pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{lr:.1e}")

    n = len(loader)
    avg_details = {k: v/n for k, v in loss_details_sum.items()}
    return epoch_loss / n, avg_details


@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device, epoch, num_epochs, use_amp):
    model.eval()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc=f"Val   [{epoch+1:>3}/{num_epochs}]", ncols=110, leave=False)

    for inputs, labels, modis, delta_days in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        modis  = modis.to(device, non_blocking=True)
        delta_days = delta_days.to(device, non_blocking=True)

        amp_ctx = (autocast(_AMP_DEVICE, enabled=use_amp)
                   if _AMP_DEVICE else autocast(enabled=use_amp))
        with amp_ctx:
            pred_hr, plru, phru = model(inputs, modis)
            loss, _ = loss_fn(pred_hr, plru, phru, labels, modis, delta_t=delta_days)

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(loader)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                             主训练函数                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="CrossScale-FPAR-Net 训练")
    parser.add_argument("--resume", action="store_true", help="从最近的 checkpoint 恢复训练")
    parser.add_argument("--test_epochs", type=int, default=0, help="仅运行 N 个 epoch 后停止 (0=禁用)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch Size")
    args = parser.parse_args()

    actual_epochs = args.test_epochs if args.test_epochs > 0 else NUM_EPOCHS
    batch_size = args.batch_size

    # ── GPU ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n[GPU] {gpu_name} ({vram:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("\n[WARN] CPU 模式")

    use_amp = USE_AMP and device.type == "cuda"

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── 数据集 ──────────────────────────────────────────────────────
    print("\n[Step 1] 构建跨尺度数据集...")
    train_ds = CrossScaleDataset(
        S1_DIR, LABEL_DIR, MODIS_DIR, DEM_PATH,
        patch_size=PATCH_SIZE, split="train",
        samples_per_image=SAMPLES_PER_IMAGE,
        hold_out_date=HOLDOUT_DATE, verbose=True,
    )
    val_ds = CrossScaleDataset(
        S1_DIR, LABEL_DIR, MODIS_DIR, DEM_PATH,
        patch_size=PATCH_SIZE, split="val",
        samples_per_image=50,
        hold_out_date=HOLDOUT_DATE, verbose=False,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            collate_fn=custom_collate)

    print(f"[数据] 训练: {len(train_ds)} patches | 验证: {len(val_ds)} patches")

    # ── 模型 ──────────────────────────────────────────────────────
    print("\n[Step 2] 初始化 CrossScale-FPAR-Net...")
    model = CrossScaleFPARNet(in_channels=IN_CHANNELS, lr_size=LR_SIZE, patch_size=PATCH_SIZE).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] 参数量: {params:.2f} M")

    loss_fn   = CrossScaleLoss(lambda_cons=0.1, lambda_temp=0.1, valid_threshold=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    # 余弦退火调度器：不看 Loss 脸色，强制按余弦曲线平滑降温
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_epochs, eta_min=1e-6)
    scaler    = (GradScaler(_AMP_DEVICE, enabled=use_amp) if _AMP_DEVICE else GradScaler(enabled=use_amp))

    best_val_loss   = float("inf")
    no_improve_cnt  = 0
    start_epoch     = 0
    best_path       = os.path.join(MODEL_SAVE_DIR, "crossscale_best_model.pth")
    ckpt_path       = os.path.join(MODEL_SAVE_DIR, "crossscale_checkpoint.pth")

    # ── Resume ────────────────────────────────────────────────────
    if args.resume and os.path.exists(ckpt_path):
        print(f"\n[Resume] 加载 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "scaler_state" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"[Resume] 从 Epoch {start_epoch} 恢复 (best_val_loss={best_val_loss:.6f})")

    # ── TensorBoard ───────────────────────────────────────────────
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    # ── 训练循环 ──────────────────────────────────────────────────
    end_epoch = start_epoch + actual_epochs
    mode_str = f"测试模式 ({actual_epochs} epochs)" if args.test_epochs > 0 else f"完整训练 ({actual_epochs} epochs)"

    print(f"\n{'═'*65}")
    print(f"  {mode_str}")
    print(f"  Batch={batch_size} | AMP={'ON' if use_amp else 'OFF'} | Resume={'ON' if args.resume else 'OFF'}")
    print(f"  LR={LEARNING_RATE:.0e} | WD=1e-2 | Warmup={WARMUP_EPOCHS}ep | Scheduler=CosineAnnealing")
    print(f"  Loss: λ_cons=0.1 | Pearson×1.5 | λ_phru=0.5")
    print(f"  Epochs: {start_epoch+1} → {end_epoch}")
    print(f"{'═'*65}\n")

    for epoch in range(start_epoch, end_epoch):
        train_loss, details = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, epoch, end_epoch, use_amp
        )
        val_loss = validate_one_epoch(
            model, val_loader, loss_fn, device, epoch, end_epoch, use_amp
        )

        # Warmup: 前 WARMUP_EPOCHS 轮线性预热
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step()
        lr = get_lr(optimizer)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalar("LR", lr, epoch + 1)
        for k, v in details.items():
            writer.add_scalar(f"Loss_Detail/{k}", v, epoch + 1)

        print(f"Epoch [{epoch+1:>3}/{end_epoch}] "
              f"train={train_loss:.5f} val={val_loss:.5f} lr={lr:.1e}"
              f" | cont={details['L_cont']:.4f} cons={details['L_cons']:.4f} phru={details['L_phru']:.4f}")

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_cnt = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "scaler_state": scaler.state_dict() if scaler else None,
                "scheduler_state": scheduler.state_dict(),
            }, best_path)
            print(f"  -> BEST model saved (val_loss={val_loss:.6f})")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n[STOP] Early stopping ({EARLY_STOP_PATIENCE} epochs no improvement)")
                break

        # Periodic checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "scaler_state": scaler.state_dict() if scaler else None,
                "scheduler_state": scheduler.state_dict(),
            }, ckpt_path)
            print(f"  [CKPT] Checkpoint 已保存 (Epoch {epoch+1})")

    writer.close()
    print(f"\n{'═'*65}")
    print(f"  训练完成! Best Val Loss = {best_val_loss:.6f}")
    print(f"  Best Model: {best_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
