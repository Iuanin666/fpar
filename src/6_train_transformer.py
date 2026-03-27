# E:\FPAR_project\src\6_train_transformer.py
# =============================================================================
# FPAR 融合模型训练脚本 —— Transformer 版本
# 核心改进：
#   1. 使用全新的 Dual-Stream Transformer 架构
#   2. 保留之前所有的 GPU 性能优化（batch=8, workers=4, persistent, cudnn, TF32）
#   3. 使用混合损失函数 (MSE + L1) 提升模型对细节的关注
# =============================================================================

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.amp import GradScaler, autocast   # PyTorch >= 2.0
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.0
    _AMP_DEVICE = None
from tqdm import tqdm

# ── 模块路径处理 ─────────────────────────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from importlib import import_module
dataset_module = import_module("2_dataset")
transformer_module = import_module("5_transformer_model")
FPAR_Fusion_Dataset = dataset_module.FPAR_Fusion_Dataset
DualStreamTransformer = transformer_module.DualStreamTransformer

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         超参数与路径配置                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

S1_DIR         = r"E:\FPAR_project\data\s1_input\aligned_output"
LABEL_DIR      = r"E:\FPAR_project\data\s2_label_aligned"
DEM_PATH       = r"E:\FPAR_project\data\s1_input\S1_Terrain_10m.tif"
MODEL_SAVE_DIR = r"E:\FPAR_project\models"
LOG_DIR        = r"E:\FPAR_project\runs_transformer"

HOLDOUT_DATE = "20250719"

IN_CHANNELS        = 7      # 空间(VV,VH,3-Terrain) + 元数据(Delta,DOY)
BATCH_SIZE         = 8      # Transformer 参数较多，显存占用大，改回 8 防止 OOM
PATCH_SIZE         = 256
SAMPLES_PER_IMAGE  = 200
NUM_EPOCHS         = 100
LEARNING_RATE      = 1e-4
LR_PATIENCE        = 10
LR_FACTOR          = 0.5
EARLY_STOP_PATIENCE = 20

# ⚠ Windows 环境下，使用 Dataset 内存缓存时禁止开启多进程，否则会因 Spawn 复制炸内存！
NUM_WORKERS = 0
USE_AMP     = True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            辅助函数                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def get_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']

# ── V5 终极修复：Huber (抵抗离群极值) + Pearson (惩罚均值退化) 混合损失 ──
class MaskedHuberPearsonLoss(nn.Module):
    def __init__(self, valid_threshold=0.05, lambda_pearson=0.5):
        super().__init__()
        self.threshold = valid_threshold
        self.lambda_pearson = lambda_pearson
        # Huber Loss: 小误差平方(平滑), 大误差绝对值(抗噪)
        self.huber = nn.HuberLoss(reduction='none', delta=0.1)

    def forward(self, preds, targets):
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        mask = (targets >= self.threshold).float()
        valid_count = mask.sum()

        if valid_count < 2:
            return (preds * 0.0).sum()

        # 1. Huber Loss
        huber_loss = (self.huber(preds, targets) * mask).sum() / valid_count

        # 2. Pearson Loss (惩罚方差崩溃)
        preds_valid = preds[mask > 0]
        targets_valid = targets[mask > 0]
        
        mean_p, std_p = preds_valid.mean(), preds_valid.std() + 1e-8
        mean_t, std_t = targets_valid.mean(), targets_valid.std() + 1e-8
        
        cov = ((preds_valid - mean_p) * (targets_valid - mean_t)).mean()
        pearson = cov / (std_p * std_t)
        pearson_loss = 1.0 - pearson

        return huber_loss + self.lambda_pearson * pearson_loss

def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device, epoch, num_epochs, use_amp
) -> float:
    model.train()
    epoch_loss = 0.0
    lr = get_lr(optimizer)

    pbar = tqdm(loader, desc=f"Train Epoch [{epoch+1:>3}/{num_epochs}]", ncols=100, leave=False)

    for inputs, targets in pbar:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (autocast(_AMP_DEVICE, enabled=use_amp)
                   if _AMP_DEVICE else autocast(enabled=use_amp))
        with amp_ctx:
            preds = model(inputs)
            loss  = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        pbar.set_postfix(loss=f"{batch_loss:.5f}", lr=f"{lr:.2e}")

    return epoch_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device, epoch, num_epochs, use_amp) -> float:
    model.eval()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc=f"Val   Epoch [{epoch+1:>3}/{num_epochs}]", ncols=100, leave=False)

    for inputs, targets in pbar:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        amp_ctx = (autocast(_AMP_DEVICE, enabled=use_amp)
                   if _AMP_DEVICE else autocast(enabled=use_amp))
        with amp_ctx:
            preds = model(inputs)
            loss  = loss_fn(preds, targets)

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.5f}")

    return epoch_loss / len(loader)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                             主训练函数                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print(f"\n[OK] GPU 优化就绪: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\n[WARN] 使用 CPU 训练")

    use_amp = USE_AMP and (device.type == "cuda")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    print("\n[步骤] 构建数据集...")
    train_dataset = FPAR_Fusion_Dataset(
        s1_dir=S1_DIR, label_dir=LABEL_DIR,
        patch_size=PATCH_SIZE, split="train",
        samples_per_image=SAMPLES_PER_IMAGE,
        hold_out_date=HOLDOUT_DATE, 
        dem_path=DEM_PATH, verbose=True
    )
    val_dataset = FPAR_Fusion_Dataset(
        s1_dir=S1_DIR, label_dir=LABEL_DIR,
        patch_size=PATCH_SIZE, split="val",
        samples_per_image=50, hold_out_date=HOLDOUT_DATE, 
        dem_path=DEM_PATH, verbose=False
    )

    _use_pin = (device.type == "cuda")
    _persist = (NUM_WORKERS > 0)
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

    print(f"[数据集] 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}\n")

    print("[步骤] 初始化 Transformer 模型...")
    model = DualStreamTransformer(in_channels=IN_CHANNELS, out_channels=1, img_size=PATCH_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] Dual-Stream Transformer 总参数量: {total_params:.2f} M\n")

    # [核心修复 V5] 使用抵抗长尾分布并且惩罚均值退化的联合 Loss
    loss_fn   = MaskedHuberPearsonLoss(valid_threshold=0.05, lambda_pearson=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )
    scaler = (GradScaler(_AMP_DEVICE, enabled=use_amp)
              if _AMP_DEVICE else GradScaler(enabled=use_amp))

    best_val_loss   = float("inf")
    no_improve_cnt  = 0
    best_model_path = os.path.join(MODEL_SAVE_DIR, "transformer_best_model.pth")

    print(f"{'═'*60}")
    print(f"  开始训练 (共 {NUM_EPOCHS} Epoch, 早停={EARLY_STOP_PATIENCE})")
    print(f"{'═'*60}\n")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, epoch, NUM_EPOCHS, use_amp
        )
        val_loss = validate_one_epoch(
            model, val_loader, loss_fn, device, epoch, NUM_EPOCHS, use_amp
        )

        scheduler.step(val_loss)
        current_lr = get_lr(optimizer)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalar("LR", current_lr, epoch + 1)

        print(
            f"Epoch [{epoch+1:>3}/{NUM_EPOCHS}]  "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_cnt = 0
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  -> ✓ 模型已保存 (Val Loss 下降至 {val_loss:.6f})")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n[!] 连续 {EARLY_STOP_PATIENCE} Epoch 无改善，触发早停。")
                break

    print(f"\n[完成] 最佳验证集 Loss = {best_val_loss:.6f}")
    print(f"[完成] 最佳模型已保存至: {best_model_path}")
    writer.close()

if __name__ == "__main__":
    main()
