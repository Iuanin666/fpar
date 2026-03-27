# E:\FPAR_project\src\4_train.py
# =============================================================================
# FPAR 融合模型训练脚本 —— 优化版
# 核心改进：
#   1. 强制检测并使用 CUDA GPU，支持 AMP 混合精度训练（节省显存、加速）
#   2. TensorBoard 记录 Train/Val Loss 和学习率曲线
#   3. tqdm 进度条实时显示 loss 和 lr
#   4. Best Model 保存（基于最低 Val Loss）
#   5. ReduceLROnPlateau 学习率自动衰减
#   6. 留一法接口：设置 HOLDOUT_DATE 将某一景排除出训练集
#   7. 空间切分验证：80% 像素训练，右下角 20% 像素验证，严格不重叠
# =============================================================================

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# PyTorch 2.x 推荐使用 torch.amp，旧版兼容 torch.cuda.amp
try:
    from torch.amp import GradScaler, autocast   # PyTorch >= 2.0
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.0
    _AMP_DEVICE = None
from tqdm import tqdm

# ── 模块路径处理（兼容直接运行或从项目根运行）──────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from importlib import import_module
dataset_module = import_module("2_dataset")
unet_module    = import_module("3_unet")
FPAR_Fusion_Dataset = dataset_module.FPAR_Fusion_Dataset
DualStreamUNet      = unet_module.DualStreamUNet


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         超参数与路径配置                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 路径配置 ──────────────────────────────────────────────────────────
S1_DIR         = r"E:\FPAR_project\data\s1_input\aligned_output" 
LABEL_DIR      = r"E:\FPAR_project\data\s2_label_aligned"
DEM_PATH       = r"E:\FPAR_project\data\s1_input\S1_Terrain_10m.tif"
MODEL_SAVE_DIR = r"E:\FPAR_project\models"
LOG_DIR        = r"E:\FPAR_project\runs"   # TensorBoard 日志目录

# ── 留一法配置 ─────────────────────────────────────────────────────────
# 设置为某景 S1 文件名中包含的日期字符串（YYYYMMDD），该景将完全排除出训练
# 设为 None 则使用全部 13 景训练
HOLDOUT_DATE = "20250719"   # 例："20250719"，留作最终测试；无留一法则改为 None

# ── 训练超参数（针对 RTX 5060 Laptop 8GB 优化）─────────────────────────
IN_CHANNELS        = 7      # 空间(VV,VH,3-Terrain) + 元数据(Delta,DOY)
BATCH_SIZE         = 16     # RTX 5060 8GB + AMP fp16 搭配 32GB 内存
                            #   ⚠ OOM 时改为 16 或 8
PATCH_SIZE         = 256    # 每次裁剪的 Patch 边长（像素）
SAMPLES_PER_IMAGE  = 200    # V2: 20→200 大幅增加训练数据量
NUM_EPOCHS         = 100    # 训练总轮数
LEARNING_RATE      = 1e-4   # 初始学习率
LR_PATIENCE        = 10     # ReduceLROnPlateau 连续多少 Epoch 无改善则降 LR
LR_FACTOR          = 0.5    # LR 衰减比例
EARLY_STOP_PATIENCE = 20    # 连续多少 Epoch Val Loss 无改善则提前停止训练

# ── 其他设置（GPU 优化）────────────────────────────────────────────────
# ⚠ 严重警告：在 Windows 下，PyTorch 多进程 (NUM_WORKERS > 0) 使用的是 spawn 模式。
# 如果我们在 Dataset 中使用巨大的 in-memory cache，会导致严重的多进程内存复制和卡死 (OOM)！
# 既然我们已经把数据全缓存在了内存里，切片提取耗时极短，单进程 (NUM_WORKERS = 0) 就能喂饱 GPU！
NUM_WORKERS = 0   
USE_AMP     = True  # 混合精度训练（fp16 前向/反向 + fp32 权重更新）


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            辅助函数                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def get_lr(optimizer: optim.Optimizer) -> float:
    """获取当前优化器的学习率（取第一个参数组）。"""
    return optimizer.param_groups[0]['lr']

# ── V5 终极修复：Huber (抵抗离群极值) + Pearson (惩罚均值退化) 混合损失 ──
class MaskedHuberPearsonLoss(nn.Module):
    def __init__(self, valid_threshold=0.05, lambda_pearson=0.5):
        super().__init__()
        self.threshold = valid_threshold
        self.lambda_pearson = lambda_pearson
        # Huber Loss 在误差小处为平方 (MSE)，在误差大处为绝对值 (L1)，对 S1 极值噪声抗性极强
        self.huber = nn.HuberLoss(reduction='none', delta=0.1)

    def forward(self, preds, targets):
        # 将 preds 展平为 (B, -1)
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        mask = (targets >= self.threshold).float()
        valid_count = mask.sum()

        if valid_count < 2:
            return (preds * 0.0).sum()

        # 1. 计算 Huber Loss
        huber_loss = (self.huber(preds, targets) * mask).sum() / valid_count

        # 2. 计算批量内的全局 Pearson Loss (强迫输出有真实的方差波动)
        preds_valid = preds[mask > 0]
        targets_valid = targets[mask > 0]
        
        # 将标准差加入极小防除零常数
        mean_p, std_p = preds_valid.mean(), preds_valid.std() + 1e-8
        mean_t, std_t = targets_valid.mean(), targets_valid.std() + 1e-8
        
        # Pearson = Cov(X,Y) / (Std(X)*Std(Y))
        # 皮尔逊系数属于 [-1, 1]。如果模型只输出常数均值，std_p 极小或退化，皮尔逊骤降。
        # 我们期望 Pearson 接近 1，所以 loss_pearson = 1 - pearson
        cov = ((preds_valid - mean_p) * (targets_valid - mean_t)).mean()
        pearson = cov / (std_p * std_t)
        pearson_loss = 1.0 - pearson

        return huber_loss + self.lambda_pearson * pearson_loss


def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device, epoch, num_epochs, use_amp
) -> float:
    """
    执行一个 Epoch 的训练，返回该 Epoch 的平均 loss。
    使用 tqdm 进度条实时显示每个 Batch 的 loss 和当前 lr。
    """
    model.train()
    epoch_loss = 0.0
    lr = get_lr(optimizer)

    pbar = tqdm(
        loader,
        desc=f"Train Epoch [{epoch+1:>3}/{num_epochs}]",
        ncols=100,
        leave=False,  # Epoch 结束后进度条自动消失，保持终端整洁
    )

    for inputs, targets in pbar:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none 比 zero_grad() 快

        # ── 前向传播（混合精度） ─────────────────────────────────────
        # autocast 参数兼容新旧 API
        amp_ctx = (autocast(_AMP_DEVICE, enabled=use_amp)
                   if _AMP_DEVICE else autocast(enabled=use_amp))
        with amp_ctx:
            preds = model(inputs)
            loss  = loss_fn(preds, targets)

        # ── 反向传播（AMP 梯度缩放） ─────────────────────────────────
        scaler.scale(loss).backward()
        
        # 梯度裁剪防爆炸 (解包后裁剪)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        # 更新进度条后缀
        pbar.set_postfix(loss=f"{batch_loss:.5f}", lr=f"{lr:.2e}")

    avg_loss = epoch_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate_one_epoch(
    model, loader, loss_fn, device, epoch, num_epochs, use_amp
) -> float:
    """
    执行一个 Epoch 的验证，返回平均 val_loss。
    不参与梯度计算，节省显存。
    """
    model.eval()
    epoch_loss = 0.0

    pbar = tqdm(
        loader,
        desc=f"Val   Epoch [{epoch+1:>3}/{num_epochs}]",
        ncols=100,
        leave=False,
    )

    for inputs, targets in pbar:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        amp_ctx = (autocast(_AMP_DEVICE, enabled=use_amp)
                   if _AMP_DEVICE else autocast(enabled=use_amp))
        with amp_ctx:
            preds = model(inputs)
            loss  = loss_fn(preds, targets)

        batch_loss = loss.item()
        epoch_loss += batch_loss
        pbar.set_postfix(loss=f"{batch_loss:.5f}")

    avg_loss = epoch_loss / len(loader)
    return avg_loss


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                             主训练函数                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    # ── 0. 设备检测与 GPU 优化 ─────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # ── GPU 性能优化 ──────────────────────────────────────────────
        # cuDNN benchmark：自动为固定尺寸输入选择最快的卷积算法
        torch.backends.cudnn.benchmark = True
        # TF32：在 Ampere+ GPU 上用 TensorFloat-32 加速矩阵运算（精度损失极小）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 设置 matmul 精度为 high（启用 TF32）
        torch.set_float32_matmul_precision('high')

        print(f"\n{'═'*60}")
        print(f"  [OK] GPU 加速已启用")
        print(f"     型号  : {gpu_name}")
        print(f"     显存  : {vram_gb:.1f} GB")
        print(f"     AMP混精度: {'开启' if USE_AMP else '关闭'}")
        print(f"     cuDNN Benchmark: 开启")
        print(f"     TF32 加速: 开启")
        print(f"     Batch Size: {BATCH_SIZE}")
        print(f"     DataLoader Workers: {NUM_WORKERS}")
        print(f"{'═'*60}\n")
    else:
        device = torch.device("cpu")
        print(f"\n{'═'*60}")
        print(f"  [WARN] 未检测到 CUDA GPU，使用 CPU 训练（速度将较慢）")
        print(f"     若确认有 GPU，请检查 CUDA 驱动和 PyTorch 版本是否匹配")
        print(f"{'═'*60}\n")

    use_amp = USE_AMP and (device.type == "cuda")  # CPU 不支持 AMP

    # ── 1. 创建目录和 TensorBoard ─────────────────────────────────────
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # TensorBoard 日志目录以时间戳命名，避免不同次训练互相覆盖
    run_name   = time.strftime("run_%Y%m%d_%H%M%S")
    writer     = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    print(f"[TensorBoard] 日志目录: {os.path.join(LOG_DIR, run_name)}")
    print(f"  → 运行命令查看: tensorboard --logdir={LOG_DIR}\n")

    # ── 2. 构建数据集 ─────────────────────────────────────────────────
    print("正在初始化数据集并匹配 S1/S2 配对...\n")

    # 训练集：空间切分左上 80%，排除 holdout 景
    train_dataset = FPAR_Fusion_Dataset(
        s1_dir=S1_DIR, label_dir=LABEL_DIR,
        patch_size=PATCH_SIZE, split="train",
        samples_per_image=SAMPLES_PER_IMAGE,
        hold_out_date=HOLDOUT_DATE,
        dem_path=DEM_PATH,  # 阶段 6: 引入地形
        verbose=True,   # 首次运行打印配对信息
    )

    # 验证集：空间切分右下 20%，同样排除 holdout 景，与训练集不重叠
    val_dataset = FPAR_Fusion_Dataset(
        s1_dir=S1_DIR, label_dir=LABEL_DIR,
        patch_size=PATCH_SIZE, split="val",
        samples_per_image=50,  # V2: 验证时每景采 50 个确定性 Patch
        hold_out_date=HOLDOUT_DATE,
        dem_path=DEM_PATH,     # 阶段 6: 引入地形
        verbose=False,  # 避免重复打印
    )

    # shuffle=True 可确保不同景的 Patch 在每个 Batch 中充分混合，
    # 避免同一景的 Patch 连续出现导致景切换时 loss 突变
    # DataLoader 优化：
    #   - pin_memory=True: 加速 CPU→GPU 数据搬运
    _use_pin = (device.type == "cuda")
    _persist = (NUM_WORKERS > 0)  # persistent_workers 仅在 num_workers>0 时有效
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=_use_pin,
        persistent_workers=_persist, prefetch_factor=4 if _persist else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=_use_pin,
        persistent_workers=_persist, prefetch_factor=4 if _persist else None,
    )

    print(f"[数据集] 训练集: {len(train_dataset)} 个 Patch | "
          f"验证集: {len(val_dataset)} 个 Patch\n")

    # ── 3. 模型、损失函数、优化器 ─────────────────────────────────────
    # [架构升级 V5] 摒弃早融合 U-Net，启用双流延迟融合网络 (Dual-Stream Late Fusion)
    model = DualStreamUNet(in_channels=IN_CHANNELS, out_channels=1).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] Dual-Stream U-Net 总参数量: {total_params:.2f} M\n")

    # [核心修复 V5] 使用抵抗长尾分布并且惩罚均值退化的联合 Loss
    loss_fn   = MaskedHuberPearsonLoss(valid_threshold=0.05, lambda_pearson=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 学习率调度器：连续 LR_PATIENCE 个 Epoch val_loss 无改善则 LR × LR_FACTOR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    # AMP GradScaler（CPU 模式下 enabled=False）
    scaler = (GradScaler(_AMP_DEVICE, enabled=use_amp)
              if _AMP_DEVICE else GradScaler(enabled=use_amp))

    # ── 4. 训练循环 ───────────────────────────────────────────────────
    best_val_loss   = float("inf")
    no_improve_cnt  = 0  # 用于 early stopping
    best_model_path = os.path.join(MODEL_SAVE_DIR, "unet_best_model.pth")

    print(f"{'═'*60}")
    print(f"  开始训练（共 {NUM_EPOCHS} Epoch）")
    print(f"  早停阈值: 连续 {EARLY_STOP_PATIENCE} Epoch 无改善则停止")
    print(f"{'═'*60}\n")

    for epoch in range(NUM_EPOCHS):
        # —— 训练 ——
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, epoch, NUM_EPOCHS, use_amp
        )

        # —— 验证 ——
        val_loss = validate_one_epoch(
            model, val_loader, loss_fn, device, epoch, NUM_EPOCHS, use_amp
        )

        # —— 学习率调度 ——
        scheduler.step(val_loss)
        current_lr = get_lr(optimizer)

        # —— TensorBoard 记录 ——
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)

        # —— 终端打印本轮摘要 ——
        print(
            f"Epoch [{epoch+1:>3}/{NUM_EPOCHS}]  "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"lr={current_lr:.2e}"
        )

        # —— Best Model 保存逻辑 ——
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            no_improve_cnt = 0
            # 保存完整 checkpoint（包含 epoch 和 optimizer state，方便续训）
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
            }, best_model_path)
            print(f"  [BEST] Best Model 已更新 → val_loss={best_val_loss:.6f}  "
                  f"[{best_model_path}]")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n  [STOP] Early Stopping 触发（连续 {EARLY_STOP_PATIENCE} Epoch 无改善）")
                break

    # ── 5. 收尾 ───────────────────────────────────────────────────────
    writer.close()
    print(f"\n{'═'*60}")
    print(f"  训练结束！")
    print(f"  最佳 Val Loss : {best_val_loss:.6f}")
    print(f"  Best Model    : {best_model_path}")
    print(f"  TensorBoard   : tensorboard --logdir={LOG_DIR}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()