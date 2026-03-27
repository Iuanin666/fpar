# E:\FPAR_project\src\7_align_modis.py
# =============================================================================
# MODIS FPAR 数据对齐脚本
# 功能：
#   1. 读取 data/2025/ 下的 MODIS 旬合成 FPAR (1000m, EPSG:32648)
#   2. 重采样至 500m
#   3. 重投影至 EPSG:4326 (与 S1/S2 一致)
#   4. 裁剪至 S1/S2 的 BoundingBox
#   5. 输出至 data/modis_aligned/
# =============================================================================

import os
import sys
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds

# ── 路径配置 ──────────────────────────────────────────────────────────────
MODIS_INPUT_DIR  = r"E:\FPAR_project\data\2025"
MODIS_OUTPUT_DIR = r"E:\FPAR_project\data\modis_aligned"
S1_REF_DIR       = r"E:\FPAR_project\data\s1_input\aligned_output"

# 目标分辨率 (度)：500m ≈ 0.0045° (在纬度 ~31°)
TARGET_RES_DEG = 500.0 / 111320.0  # ~0.00449°
TARGET_CRS = "EPSG:4326"


def get_s1_bounds(s1_dir: str):
    """从 S1 参考文件获取目标空间范围 (EPSG:4326)。"""
    s1_files = [f for f in os.listdir(s1_dir) if f.lower().endswith('.tif')]
    if not s1_files:
        raise FileNotFoundError(f"S1 目录中无 TIFF 文件: {s1_dir}")
    ref_path = os.path.join(s1_dir, s1_files[0])
    with rasterio.open(ref_path) as src:
        return src.bounds, src.crs


def align_single_modis(input_path: str, output_path: str, target_bounds, target_crs: str):
    """
    将单个 MODIS TIFF 重采样至 500m 并重投影裁剪。
    """
    with rasterio.open(input_path) as src:
        # 1. 计算从源 CRS -> 目标 CRS 的变换
        transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            src.width, src.height,
            *src.bounds,
            resolution=TARGET_RES_DEG,
        )

        # 2. 创建重投影后的临时数组
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'count': 1,
        })

        # 重投影到内存
        reprojected = np.zeros((1, height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    # 3. 裁剪至 S1/S2 的 BoundingBox
    from rasterio.transform import from_bounds as transform_from_bounds
    
    # 计算裁剪窗口 (像素坐标)
    left, bottom, right, top = target_bounds

    # 将地理坐标转为像素坐标
    inv_transform = ~transform
    col_start, row_start = inv_transform * (left, top)
    col_end, row_end = inv_transform * (right, bottom)

    row_start = max(0, int(np.floor(row_start)))
    row_end   = min(height, int(np.ceil(row_end)))
    col_start = max(0, int(np.floor(col_start)))
    col_end   = min(width, int(np.ceil(col_end)))

    clipped = reprojected[:, row_start:row_end, col_start:col_end]
    clip_h, clip_w = clipped.shape[1], clipped.shape[2]

    # 新的仿射变换 (裁剪后的左上角)
    new_origin_x = transform.c + col_start * transform.a
    new_origin_y = transform.f + row_start * transform.e
    clip_transform = rasterio.transform.Affine(
        transform.a, transform.b, new_origin_x,
        transform.d, transform.e, new_origin_y,
    )

    # 4. 写出
    out_meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': clip_w,
        'height': clip_h,
        'count': 1,
        'crs': target_crs,
        'transform': clip_transform,
        'compress': 'lzw',
    }

    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(clipped)

    return clip_h, clip_w


def main():
    print("=" * 60)
    print("  MODIS FPAR 数据对齐脚本")
    print("  1000m → 500m，EPSG:32648 → EPSG:4326")
    print("=" * 60)

    # 获取 S1 参考范围
    s1_bounds, s1_crs = get_s1_bounds(S1_REF_DIR)
    print(f"\n[参考] S1 Bounds (EPSG:4326): {s1_bounds}")
    print(f"[参考] S1 CRS: {s1_crs}")
    print(f"[目标] 分辨率: ~500m ({TARGET_RES_DEG:.6f}°)")

    # 创建输出目录
    os.makedirs(MODIS_OUTPUT_DIR, exist_ok=True)

    # 处理所有 MODIS 文件
    modis_files = sorted([f for f in os.listdir(MODIS_INPUT_DIR) if f.endswith('.tif')])
    print(f"\n[数据] 发现 {len(modis_files)} 个 MODIS TIFF 文件")

    for i, fname in enumerate(modis_files):
        input_path  = os.path.join(MODIS_INPUT_DIR, fname)
        # 保持文件名，添加 _500m 后缀
        base = os.path.splitext(fname)[0]
        output_path = os.path.join(MODIS_OUTPUT_DIR, f"{base}_500m.tif")

        h, w = align_single_modis(input_path, output_path, s1_bounds, TARGET_CRS)
        print(f"  [{i+1}/{len(modis_files)}] {fname} → {os.path.basename(output_path)} ({h}×{w})")

    # 验证输出
    print(f"\n[完成] 所有文件已保存至: {MODIS_OUTPUT_DIR}")
    
    # 打印最终对齐结果
    out_files = sorted([f for f in os.listdir(MODIS_OUTPUT_DIR) if f.endswith('.tif')])
    if out_files:
        ref_path = os.path.join(MODIS_OUTPUT_DIR, out_files[0])
        with rasterio.open(ref_path) as src:
            print(f"\n[验证] 对齐后参数:")
            print(f"  CRS: {src.crs}")
            print(f"  Shape: {src.shape}")
            print(f"  Resolution: {src.res}")
            print(f"  Bounds: {src.bounds}")
    print("=" * 60)


if __name__ == "__main__":
    main()
