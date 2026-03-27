# E:\FPAR_project\src\1_align_data.py
import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from tqdm import tqdm


def align_s2_to_s1_10m(s1_reference_path, s2_path, output_path):
    """将 S2 (20m) 升尺度并严格对齐到 S1 (10m) 的网格上"""
    with rasterio.open(s1_reference_path) as src_s1, rasterio.open(s2_path) as src_s2:
        # 获取 S2 的波段数
        s2_bands = src_s2.count

        # 创建一个空数组，形状匹配 S1 的宽、高 (10m)，波段数匹配 S2
        s2_resampled = np.zeros((s2_bands, src_s1.height, src_s1.width), dtype=src_s2.dtypes[0])

        # 遍历重采样 (FPAR 是连续值，必须用双线性插值 bilinear)
        for i in range(1, s2_bands + 1):
            reproject(
                source=rasterio.band(src_s2, i),
                destination=s2_resampled[i - 1],
                src_transform=src_s2.transform,
                src_crs=src_s2.crs,
                dst_transform=src_s1.transform,
                dst_crs=src_s1.crs,
                resampling=Resampling.bilinear
            )

        # 准备输出的元数据：借用 S1 的空间信息(10m)，保留 S2 的数据类型
        profile = src_s1.profile
        profile.update(
            dtype=src_s2.dtypes[0],
            count=s2_bands,
            nodata=src_s2.nodata
        )

        # 保存 10m 版本的 S2
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(s2_resampled)


if __name__ == "__main__":
    # 配置路径
    s1_dir = r"E:\FPAR_project\data\s1_input"  # 基准：10m 的 S1
    s2_label_dir = r"E:\FPAR_project\data\s2_lable"  # 待处理：20m 的 S2 FPAR
    out_label_dir = r"E:\FPAR_project\data\s2_label_10m"  # 输出：10m 的 S2 FPAR

    os.makedirs(out_label_dir, exist_ok=True)

    s1_files = sorted([f for f in os.listdir(s1_dir) if f.endswith('.tif')])
    s2_label_files = sorted([f for f in os.listdir(s2_label_dir) if f.endswith('.tif')])

    print("开始将 S2(20m) 升尺度对齐至 S1(10m)...")
    # 假设文件是一一对应的 (如果文件名不同，请确保它们按时间顺序排好是对得上的)
    for s1_file, s2_file in tqdm(zip(s1_files, s2_label_files), total=len(s1_files)):
        s1_path = os.path.join(s1_dir, s1_file)
        s2_path = os.path.join(s2_label_dir, s2_file)
        out_path = os.path.join(out_label_dir, s2_file.replace('.tif', '_10m.tif'))

        align_s2_to_s1_10m(s1_path, s2_path, out_path)

    print(f"对齐完成！所有 S2 标签(10m) 数据已保存在: {out_label_dir}")