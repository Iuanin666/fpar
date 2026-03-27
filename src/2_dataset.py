# E:\FPAR_project\src\2_dataset.py
# =============================================================================
# FPAR 融合数据集 —— V2 修复版
# 核心改进：
#   1. 自动解析 S1/S2 文件名中的日期，最近邻时间匹配配对
#   2. 空间切分验证：左上 80% 训练 / 右下 20% 验证
#   3. 留一法接口：hold_out_date 排除指定日期
#   4. 多 Patch 采样：默认每景 200 个 Patch
#   5. [修复] S1 归一化：per-channel min-max clamp 替代错误的 /30
#   6. [新增] 无效 Patch 过滤：有效 FPAR 像素 < 30% 的 Patch 自动跳过
#   7. [新增] 数据增强：随机水平/垂直翻转 + 随机 90° 旋转
# =============================================================================

import os
import re
import random
from datetime import datetime

import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：从文件名解析日期
# ──────────────────────────────────────────────────────────────────────────────

def _parse_s1_date(filename: str) -> datetime:
    """
    解析 S1 文件名中的日期。
    例：S1A_IW_GRDH_1SDV_20250402T110058_... → datetime(2025, 4, 2)
    使用正则匹配形如 _YYYYMMDT 的日期字段（第 5 段下划线字段）。
    """
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法从 S1 文件名中解析日期: {filename}")


def _parse_s2_date(filename: str) -> datetime:
    """
    解析 S2 文件名中的日期。
    例：S2_FPAR_Mosaic_20250427_10m.tif → datetime(2025, 4, 27)
    """
    match = re.search(r'_(\d{8})_', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    raise ValueError(f"无法从 S2 文件名中解析日期: {filename}")


def _get_s1_date_str(filename: str) -> str:
    """
    返回 S1 文件名中的纯日期字符串 (YYYYMMDD)，用于 hold_out_date 比对。
    """
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        return match.group(1)
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 类
# ──────────────────────────────────────────────────────────────────────────────

class FPAR_Fusion_Dataset(Dataset):
    """
    FPAR 多源遥感融合数据集。

    参数
    ----
    s1_dir : str
        Sentinel-1 TIFF 文件目录（VV/VH 双通道，10m）。
    label_dir : str
        Sentinel-2 FPAR TIFF 文件目录（单通道，已配准到 10m）。
    patch_size : int
        每次裁剪的图像块边长（像素），默认 256。
    split : str
        "train"  → 从每景影像左上角 80%×80% 的区域内随机采 Patch；
        "val"    → 从每景影像右下角 20% 区域的中心采 1 个 Patch（与训练区不重叠）；
        "test"   → 用于外部评估，行为同 "val"，但通常配合 hold_out_date 使用。
    samples_per_image : int
        训练时每景采多少个随机 Patch（验证/测试时固定为 1）。
    hold_out_date : str or None
        "YYYYMMDD" 格式的日期字符串（如 "20250719"）。
        含有该日期的 S1 景及其配对 S2 将被当前 Dataset 完全排除，
        供留一法交叉验证中的 Test Set 单独使用。
    verbose : bool
        是否在初始化时打印配对信息（建议首次运行时开启）。
    """

    def __init__(
        self,
        s1_dir: str,
        label_dir: str,
        patch_size: int = 256,
        split: str = "train",          # "train" | "val" | "test"
        samples_per_image: int = 200,   # V2: 20→200 大幅增加数据量
        hold_out_date: str = None,
        verbose: bool = True,
        min_valid_ratio: float = 0.3,   # 有效 FPAR 像素最低占比
        dem_path: str = None,          # 阶段 6: DEM 路径
    ):
        super().__init__()
        assert split in ("train", "val", "test"), \
            f"split 参数必须为 'train'、'val' 或 'test'，当前值: {split}"

        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.split = split
        self.samples_per_image = samples_per_image
        self.hold_out_date = hold_out_date
        self.min_valid_ratio = min_valid_ratio
        
        # ── 0. 预加载静态 DEM ──
        self.dem_data = None
        self.dem_min = []
        self.dem_max = []
        if dem_path and os.path.exists(dem_path):
            with rasterio.open(dem_path) as src:
                # 阶段 7: 读取前 3 个波段 (海拔, 坡度, 坡向)
                self.dem_data = src.read((1, 2, 3)).astype(np.float32)
                # 计算全局 Min-Max 用于后续自适应归一化，不再死算假设值
                for i in range(3):
                    self.dem_min.append(np.nanmin(self.dem_data[i]))
                    self.dem_max.append(np.nanmax(self.dem_data[i]))
            if verbose:
                print(f"[数据] DEM 载入成功: {os.path.basename(dem_path)} (Shape: {self.dem_data.shape})")
                for i, name in enumerate(["海拔(B1)", "坡度(B2)", "坡向(B3)"]):
                    print(f"  - {name}: Min={self.dem_min[i]:.2f}, Max={self.dem_max[i]:.2f}")

        # ── 1. 获取并排序文件列表（S1 和 S2 数量可以不同）──────────────
        s1_files = sorted([f for f in os.listdir(s1_dir)    if f.lower().endswith('.tif')])
        s2_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.tif')])

        # 解析 S2 文件的日期，用于后续最近邻搜索
        s2_dates = []
        for s2f in s2_files:
            try:
                s2_dates.append(_parse_s2_date(s2f))
            except ValueError:
                s2_dates.append(None)

        # ── 2. 最近邻时间匹配：每个 S1 找时间差最小的 S2 ─────────────
        # 允许多对一（多个 S1 映射到同一个 S2），不要求数量相等
        raw_pairs = []
        for s1f in s1_files:
            try:
                d1 = _parse_s1_date(s1f)
            except ValueError as e:
                print(f"[警告] S1 日期解析失败，已跳过: {e}")
                continue

            # 找时间差最小的 S2
            best_idx, best_delta = -1, float("inf")
            for j, (s2f, d2) in enumerate(zip(s2_files, s2_dates)):
                if d2 is None:
                    continue
                delta = abs((d1 - d2).days)
                if delta < best_delta:
                    best_delta = delta
                    best_idx = j

            if best_idx < 0:
                print(f"[警告] 找不到 S1={d1.date()} 的配对 S2，已跳过")
                continue

            # 参考 SenRVM，取消硬性 5 天对齐，允许一切时间差，交由模型作为特征去学习跨越时间的植被动态
            # 仅做极端限高（>180天视为不同季节完全无意义，剔除）
            if best_delta > 180:
                print(f"[警告] S1={d1.date()} 的最佳配对 S2={s2_dates[best_idx].date()} 差 {best_delta} 天，完全不同季，丢弃")
                continue

            raw_pairs.append((
                s1f, s2_files[best_idx],
                d1, s2_dates[best_idx], best_delta
            ))

        if verbose:
            print("\n" + "═" * 72)
            print(f"{'序号':>4}  {'S1 日期':^10}  {'S2 日期':^10}  {'时间差(天)':^10}  状态")
            print("─" * 72)
            for i, (s1f, s2f, d1, d2, delta) in enumerate(raw_pairs):
                # 均标记为可用，由神经网络通过第三通道自行理解时间流逝
                flag = f"[Y] 差 {delta} 天"
                hout = " [HOLDOUT]" if (hold_out_date and hold_out_date in s1f) else ""
                print(f"  {i+1:>2}  {d1.strftime('%Y-%m-%d'):^10}  "
                      f"{d2.strftime('%Y-%m-%d'):^10}  {delta:^10}  {flag}{hout}")
            print("═" * 72 + "\n")

        # ── 3. 根据 hold_out_date 过滤配对 ───────────────────────────
        self.pairs = []
        self.holdout_pairs = []
        for pair in raw_pairs:
            s1f = pair[0]
            if hold_out_date and hold_out_date in s1f:
                self.holdout_pairs.append(pair)
            else:
                self.pairs.append(pair)

        if verbose:
            print(f"[数据集 split='{split}'] 有效训练对: {len(self.pairs)} 景"
                  f"，排除 holdout: {len(self.holdout_pairs)} 景")

        if len(self.pairs) == 0:
            raise RuntimeError("有效配对数量为 0，请检查 hold_out_date 或数据文件。")

        # ── 4. 内存级缓存字典 ──────────────────────────────────────────────
        # 缓存整幅对齐后的 S1 和 S2 数据，打穿磁盘 IO 瓶颈
        self.cache = {}

    # ──────────────────────────────────────────────────────────────────
    def __len__(self):
        # 训练和验证均按 samples_per_image 采多个 Patch，充分利用数据
        return len(self.pairs) * self.samples_per_image

    # ──────────────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        # 所有 split 统一使用 samples_per_image 计算景索引
        scene_idx = idx // self.samples_per_image

        s1_fname, s2_fname, *_ = self.pairs[scene_idx]
        s1_path = os.path.join(self.s1_dir, s1_fname)
        label_path = os.path.join(self.label_dir, s2_fname)

        # ── 内存缓存机制 (极速提升读取速度) ──────────────────────────
        if scene_idx not in self.cache:
            # 读取 TIFF 数据
            with rasterio.open(s1_path) as src:
                s1_data = src.read().astype(np.float32)   # (C, H, W)

            with rasterio.open(label_path) as src:
                label_data = src.read(1).astype(np.float32)  # (H, W)

            # 对齐尺寸（阶段 6: 执行 S1, S2, DEM 的三路严格对齐）
            h_list = [s1_data.shape[1], label_data.shape[0]]
            w_list = [s1_data.shape[2], label_data.shape[1]]
            if self.dem_data is not None:
                h_list.append(self.dem_data.shape[1])
                w_list.append(self.dem_data.shape[2])
            
            min_h = min(h_list)
            min_w = min(w_list)
            
            s1_data = s1_data[:, :min_h, :min_w]
            label_data = label_data[:min_h, :min_w]
            
            # 如果有 DEM，也进行同步切边对齐
            if self.dem_data is not None:
                dem_aligned = self.dem_data[:, :min_h, :min_w]
            else:
                dem_aligned = np.zeros((3, min_h, min_w), dtype=np.float32)
            
            self.cache[scene_idx] = (s1_data, label_data, dem_aligned)
        else:
            s1_data, label_data, dem_aligned = self.cache[scene_idx]

        _, H, W = s1_data.shape
        ps = self.patch_size

        # ── [核心修复 V5] 空间分块验证 (Spatial Block Cross-Validation) ─────────
        # 为了彻底杜绝空间自相关性导致的数据泄漏 (Data Leakage)，我们抛弃随机的左上/右下像素划分。
        # 强制将整张遥感图像切分为 5 x 5 = 25 个固定的宏观区块 (Macro-blocks)。
        # 前 20 个区块 (80%) 专属于训练集，后 5 个区块 (20%) 专属于验证集，中间有绝对的物理边界。
        
        block_rows = 5
        block_cols = 5
        block_h = H // block_rows
        block_w = W // block_cols
        
        if self.split == "train":
            # 训练集：随机选择前 20 个区块 (索引 0~19) 中的一个
            block_idx = random.randint(0, 19)
        else:
            # 验证集：确定性选择后 5 个区块 (索引 20~24) 中的一个
            rng = random.Random(idx)
            block_idx = rng.randint(20, 24)
            
        # 计算当前选中区块的左上角坐标范围
        b_r = block_idx // block_cols
        b_c = block_idx % block_cols
        
        b_top_start = b_r * block_h
        b_left_start = b_c * block_w
        
        # 定义当前区块的有效裁剪极值 (确保 Patch 不跨越区块边界，从而100%隔离)
        h_max_top_block = max(b_top_start, b_top_start + block_h - ps)
        w_max_left_block = max(b_left_start, b_left_start + block_w - ps)
        
        if self.split == "train":
            top = random.randint(b_top_start, h_max_top_block)
            left = random.randint(b_left_start, w_max_left_block)
        else:
            top = rng.randint(b_top_start, h_max_top_block)
            left = rng.randint(b_left_start, w_max_left_block)

        # 最终安全越界保护
        top = min(top, max(0, H - ps))
        left = min(left, max(0, W - ps))

        # ── 裁剪 Patch ────────────────────────────────────────────────
        s1_patch = s1_data[:, top:top + ps, left:left + ps]
        label_patch = label_data[top:top + ps, left:left + ps]
        dem_patch = dem_aligned[:, top:top + ps, left:left + ps]

        # ── 无效 Patch 过滤 ────────────────────────────────────────────
        # 如果有效 FPAR 像素 (>= 0.05) 不足 min_valid_ratio，则在同区块内重新采样
        valid_ratio = np.sum(label_patch >= 0.05) / label_patch.size
        if self.split == "train" and valid_ratio < self.min_valid_ratio:
            # 用另一个随机位置重试（最多 5 次）
            for _ in range(5):
                top2 = random.randint(b_top_start, h_max_top_block)
                left2 = random.randint(b_left_start, w_max_left_block)
                top2 = min(top2, max(0, H - ps))
                left2 = min(left2, max(0, W - ps))
                label_patch = label_data[top2:top2 + ps, left2:left2 + ps]
                if np.sum(label_patch >= 0.05) / label_patch.size >= self.min_valid_ratio:
                    s1_patch = s1_data[:, top2:top2 + ps, left2:left2 + ps]
                    dem_patch = dem_aligned[:, top2:top2 + ps, left2:left2 + ps]
                    break
            else:
                s1_patch = s1_data[:, top2:top2 + ps, left2:left2 + ps]
                dem_patch = dem_aligned[:, top2:top2 + ps, left2:left2 + ps]

        # ── 转 Tensor ─────────────────────────────────────────────────
        input_tensor = torch.from_numpy(s1_patch.copy()).float()
        label_tensor = torch.from_numpy(label_patch.copy()).float().unsqueeze(0)  # (1, H, W)

        # ── 清洗异常值（NaN / Inf → 0） ──────────────────────────────
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        label_tensor = torch.nan_to_num(label_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # ── S1 归一化 (V2 修复) ────────────────────────────────────────
        # 实测 S1 dB 范围: VV [-42, +14], VH [-34, +24]
        # 使用 per-channel clamp + rescale 到 [0, 1]
        VV_MIN, VV_MAX = -30.0, 5.0
        VH_MIN, VH_MAX = -35.0, 5.0
        input_tensor[0] = (input_tensor[0].clamp(VV_MIN, VV_MAX) - VV_MIN) / (VV_MAX - VV_MIN)
        input_tensor[1] = (input_tensor[1].clamp(VH_MIN, VH_MAX) - VH_MIN) / (VH_MAX - VH_MIN)

        # ── 构建元数据通道 (DeltaT + DOY) ───────────────────────────
        # 1. 有符号时间差：S1 日期 - S2 日期 (归一化到约 [-1, 1])
        d1, d2, _ = self.pairs[scene_idx][2], self.pairs[scene_idx][3], self.pairs[scene_idx][4]
        delta_norm = (d1 - d2).days / 30.0
        delta_channel = torch.full((1, ps, ps), delta_norm, dtype=torch.float32)

        # 2. DOY (一年中的第几天) - 提供强力的季节先验 (归一化到 [0, 1])
        doy_norm = float(d1.timetuple().tm_yday) / 366.0
        doy_channel = torch.full((1, ps, ps), doy_norm, dtype=torch.float32)

        # ── 拼接最终 7 通道输入: (VV, VH, Elevation, Slope, Aspect, DeltaT, DOY) ──
        # 1. 地形通道自适应归一化 (根据全局 Min-Max)
        dem_norm_patches = []
        if self.dem_data is not None:
            for i in range(3):
                p = dem_patch[i]
                denom = self.dem_max[i] - self.dem_min[i]
                p_norm = (p - self.dem_min[i]) / (denom if denom > 1e-6 else 1.0)
                dem_norm_patches.append(torch.from_numpy(p_norm[np.newaxis].copy()).float())
        else:
            for i in range(3):
                dem_norm_patches.append(torch.zeros((1, ps, ps), dtype=torch.float32))

        input_tensor = torch.cat([
            input_tensor,           # (2, ps, ps) - VV, VH
            *dem_norm_patches,      # (3, ps, ps) - Elevation, Slope, Aspect
            delta_channel,          # (1, ps, ps) - DeltaT
            doy_channel             # (1, ps, ps) - DOY
        ], dim=0)

        # FPAR 物理范围 0~1，严格强制截断 <0.05 超范围的底噪值以免毒害模型
        label_tensor[label_tensor < 0.05] = 0.0
        label_tensor = torch.clamp(label_tensor, 0.0, 1.0)

        # ── 数据增强（仅训练集）─────────────────────────────────────────
        if self.split == "train":
            # 随机水平翻转
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [2])  # 沿 W 轴
                label_tensor = torch.flip(label_tensor, [2])
            # 随机垂直翻转
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [1])  # 沿 H 轴
                label_tensor = torch.flip(label_tensor, [1])
            # 随机 90° 旋转 (k=0,1,2,3)
            k = random.randint(0, 3)
            if k > 0:
                input_tensor = torch.rot90(input_tensor, k, [1, 2])
                label_tensor = torch.rot90(label_tensor, k, [1, 2])

        return input_tensor, label_tensor

    # ──────────────────────────────────────────────────────────────────
    def get_holdout_dataset(self, patch_size=None):
        """
        返回一个包含 holdout 景的简单 Dataset 对象，用于最终测试评估。
        （在 evaluate.py 中使用，不在训练循环中调用）
        """
        if not self.holdout_pairs:
            raise RuntimeError("当前 Dataset 没有配置 hold_out_date 或未找到对应景。")
        return _HoldoutDataset(
            pairs=self.holdout_pairs,
            s1_dir=self.s1_dir,
            label_dir=self.label_dir,
            patch_size=patch_size or self.patch_size,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 内部辅助类：HoldoutDataset（每景返回 1 个中心 Patch，供快速评估）
# ──────────────────────────────────────────────────────────────────────────────

class _HoldoutDataset(Dataset):
    """不对外使用，仅由 FPAR_Fusion_Dataset.get_holdout_dataset() 创建。"""

    def __init__(self, pairs, s1_dir, label_dir, patch_size=256):
        self.pairs = pairs
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.patch_size = patch_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1_fname, s2_fname, *_ = self.pairs[idx]
        s1_path = os.path.join(self.s1_dir, s1_fname)
        label_path = os.path.join(self.label_dir, s2_fname)

        with rasterio.open(s1_path) as src:
            s1_data = src.read().astype(np.float32)
        with rasterio.open(label_path) as src:
            label_data = src.read(1).astype(np.float32)

        min_h = min(s1_data.shape[1], label_data.shape[0])
        min_w = min(s1_data.shape[2], label_data.shape[1])
        s1_data = s1_data[:, :min_h, :min_w]
        label_data = label_data[:min_h, :min_w]

        _, H, W = s1_data.shape
        ps = self.patch_size
        top = max(0, (H - ps) // 2)
        left = max(0, (W - ps) // 2)

        s1_patch = s1_data[:, top:top + ps, left:left + ps]
        label_patch = label_data[top:top + ps, left:left + ps]

        input_tensor = torch.from_numpy(s1_patch.copy()).float()
        label_tensor = torch.from_numpy(label_patch.copy()).float().unsqueeze(0)
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        label_tensor = torch.nan_to_num(label_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        # V2 修复：与主 Dataset 相同的 per-channel min-max 归一化
        VV_MIN, VV_MAX = -30.0, 5.0
        VH_MIN, VH_MAX = -35.0, 5.0
        input_tensor[0] = (input_tensor[0].clamp(VV_MIN, VV_MAX) - VV_MIN) / (VV_MAX - VV_MIN)
        input_tensor[1] = (input_tensor[1].clamp(VH_MIN, VH_MAX) - VH_MIN) / (VH_MAX - VH_MIN)
        label_tensor = torch.clamp(label_tensor, 0.0, 1.0)
        return input_tensor, label_tensor