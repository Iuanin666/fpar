# E:\FPAR_project\src\8_crossscale_model.py
# =============================================================================
# CrossScale-FPAR-Net: 双向拼接对比学习架构
# Bi-directional Patch Contrastive Learning for Cross-Scale FPAR Inversion
#
# 输入三路:
#   - S1 SAR 引导 (VV/VH + Terrain + Meta) : 7ch @ 256×256
#   - S2 HRU 真值 (FPAR)                    : 1ch @ 256×256 (训练时)
#   - MODIS LRU 先验 (FPAR)                 : 1ch @ 5×5
#
# 核心模块:
#   1. HR Encoder:       编码 S1 空间纹理
#   2. Aggregation:      高→低拼接 (生成 PLRU)
#   3. Disaggregation:   低→高拆解 (生成 PHRU)
#   4. Patch Embedding:  多源特征空间对齐
#   5. Dual-Stream Transformer: 学习混合与还原映射
#   6. Decoder:          输出 10m FPAR
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# 基础构建块
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """双层卷积块 (Conv-BN-GELU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """通道注意力 (SE Block)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


# ══════════════════════════════════════════════════════════════════════════════
# 1. HR Encoder: 编码 S1 SAR 空间纹理
# ══════════════════════════════════════════════════════════════════════════════

class HREncoder(nn.Module):
    """
    多尺度编码器,提取 S1 SAR + Terrain 的空间纹理特征。
    输入: (B, 7, 256, 256)
    输出: 多层特征 [f1(64,256,256), f2(128,128,128), f3(256,64,64), f4(512,32,32)]
    """
    def __init__(self, in_channels=7):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.enc1(x)               # (B, 64, 256, 256)
        f2 = self.enc2(self.pool(f1))    # (B, 128, 128, 128)
        f3 = self.enc3(self.pool(f2))    # (B, 256, 64, 64)
        f4 = self.enc4(self.pool(f3))    # (B, 512, 32, 32)
        return [f1, f2, f3, f4]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Aggregation Module: 高→低拼接 (HRU → PLRU)
# ══════════════════════════════════════════════════════════════════════════════

class AggregationModule(nn.Module):
    """
    将高分特征聚合为低分伪 FPAR (Pseudo-LRU / PLRU)。
    通过自适应池化将 HR 特征图降至 MODIS 尺度 (5×5)，
    然后用 1×1 卷积回归出 PLRU FPAR 值。
    """
    def __init__(self, hr_channels=512, lr_size=5):
        super().__init__()
        self.lr_size = lr_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(lr_size)
        self.regressor = nn.Sequential(
            nn.Conv2d(hr_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, hr_feat):
        # hr_feat: (B, 512, 32, 32)
        pooled = self.adaptive_pool(hr_feat)  # (B, 512, 5, 5)
        plru = self.regressor(pooled)          # (B, 1, 5, 5)
        return plru


# ══════════════════════════════════════════════════════════════════════════════
# 3. Disaggregation Module: 低→高拆解 (LRU → PHRU)
# ══════════════════════════════════════════════════════════════════════════════

class DisaggregationModule(nn.Module):
    """
    将低分 MODIS FPAR 拆解为高分伪 FPAR (Pseudo-HRU / PHRU)。
    使用 SAR 纹理引导的空间注意力 + 逐级上采样。
    """
    def __init__(self, guide_channels=512, target_size=256):
        super().__init__()
        self.target_size = target_size

        # MODIS 特征提取
        self.lr_embed = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # SAR 引导的空间注意力
        self.guide_attn = nn.Sequential(
            nn.Conv2d(guide_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 1),
            nn.Sigmoid(),
        )

        # 逐级上采样并融合
        self.fuse_and_up = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, modis_lr, hr_feat):
        """
        modis_lr: (B, 1, lr_h, lr_w) - 原始 MODIS FPAR
        hr_feat:  (B, 512, 32, 32)   - HR Encoder 最深层特征
        """
        # 上采样 MODIS 到 HR Encoder 特征图尺度
        _, _, fh, fw = hr_feat.shape
        lr_up = F.interpolate(modis_lr, size=(fh, fw), mode='bilinear', align_corners=False)
        lr_feat = self.lr_embed(lr_up)  # (B, 128, 32, 32)

        # SAR 纹理引导注意力
        attn = self.guide_attn(hr_feat)  # (B, 128, 32, 32)
        guided = lr_feat * attn           # (B, 128, 32, 32)

        # 拼接 + 融合
        fused = torch.cat([guided, lr_feat], dim=1)  # (B, 256, 32, 32)
        phru_small = self.fuse_and_up(fused)           # (B, 1, 32, 32)

        # 上采样到最终分辨率
        phru = F.interpolate(phru_small, size=(self.target_size, self.target_size),
                             mode='bilinear', align_corners=False)
        return phru  # (B, 1, 256, 256)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dual-Stream Transformer
# ══════════════════════════════════════════════════════════════════════════════

class CrossScaleTransformer(nn.Module):
    """
    跨尺度 Transformer: 在 Bottleneck 层融合 HR 特征和 LR 先验。
    """
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, ff_dim=1024):
        super().__init__()
        # LR 特征投影到 embed_dim
        self.lr_proj = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 通道注意力融合
        self.fuse_attn = ChannelAttention(embed_dim * 2)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, hr_feat, modis_lr):
        """
        hr_feat:  (B, 512, 32, 32) - HR Encoder bottleneck
        modis_lr: (B, 1, lr_h, lr_w) - MODIS FPAR
        """
        B, C, H, W = hr_feat.shape

        # 上采样 MODIS 到 bottleneck 尺度
        lr_up = F.interpolate(modis_lr, size=(H, W), mode='bilinear', align_corners=False)
        lr_feat = self.lr_proj(lr_up)  # (B, 512, 32, 32)

        # 拼接 HR + LR 特征
        combined = torch.cat([hr_feat, lr_feat], dim=1)  # (B, 1024, 32, 32)
        combined = self.fuse_attn(combined)
        fused = self.fuse_conv(combined)  # (B, 512, 32, 32)

        # Transformer 处理 (flatten spatial dims)
        tokens = fused.flatten(2).transpose(1, 2)  # (B, H*W, 512)
        tokens = self.transformer(tokens)            # (B, H*W, 512)
        out = tokens.transpose(1, 2).view(B, C, H, W)  # (B, 512, 32, 32)

        return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. Decoder
# ══════════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    U-Net 式解码器,带 Skip Connection。
    输入: bottleneck (512, 32, 32) + encoder features
    输出: (1, 256, 256)
    """
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)  # 256 + 256 skip

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)  # 128 + 128 skip

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)   # 64 + 64 skip

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, bottleneck, skip_features):
        f1, f2, f3 = skip_features  # 64/256, 128/128, 256/64

        d3 = self.up3(bottleneck)            # (B, 256, 64, 64)
        d3 = torch.cat([d3, f3], dim=1)      # (B, 512, 64, 64)
        d3 = self.dec3(d3)                    # (B, 256, 64, 64)

        d2 = self.up2(d3)                     # (B, 128, 128, 128)
        d2 = torch.cat([d2, f2], dim=1)       # (B, 256, 128, 128)
        d2 = self.dec2(d2)                    # (B, 128, 128, 128)

        d1 = self.up1(d2)                     # (B, 64, 256, 256)
        d1 = torch.cat([d1, f1], dim=1)       # (B, 128, 256, 256)
        d1 = self.dec1(d1)                    # (B, 64, 256, 256)

        out = self.final(d1)                  # (B, 1, 256, 256)
        return torch.sigmoid(out)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 完整模型: CrossScaleFPARNet
# ══════════════════════════════════════════════════════════════════════════════

class CrossScaleFPARNet(nn.Module):
    """
    双向拼接对比学习 FPAR 反演网络

    训练时返回: (pred_hr, plru, phru)
    推理时返回: pred_hr
    """
    def __init__(self, in_channels=8, lr_size=5, patch_size=256):
        super().__init__()
        self.patch_size = patch_size

        # 核心模块
        self.hr_encoder = HREncoder(in_channels)
        self.aggregation = AggregationModule(hr_channels=512, lr_size=lr_size)
        self.disaggregation = DisaggregationModule(guide_channels=512, target_size=patch_size)
        self.cross_transformer = CrossScaleTransformer(embed_dim=512)
        self.decoder = Decoder()

    def forward(self, s1_input, modis_lr=None):
        """
        s1_input: (B, 7, 256, 256) - S1 SAR + Terrain + Meta
        modis_lr: (B, 1, lr_h, lr_w) - MODIS FPAR (仅用于训练时的知识蒸馏监督)
        """
        # 1. HR Encoder
        f1, f2, f3, f4 = self.hr_encoder(s1_input)

        # 2. Aggregation: 高→低 (生成 PLRU)
        plru = self.aggregation(f4)  # (B, 1, 5, 5)

        # 3. Disaggregation: 低→高 (生成 PHRU)
        if modis_lr is not None:
            # 训练时：用真实 MODIS 测试拆解能力，通过 L_phru 约束学习
            phru = self.disaggregation(modis_lr, f4)  # (B, 1, 256, 256)
        else:
            # 推理时：使用自己生成的 PLRU 拆解
            phru = self.disaggregation(plru, f4)

        # 4. CrossScale Transformer [核心修复！]
        # 无论训练还是推理，统一使用自己生成的 plru。
        # 断绝模型对真实 MODIS 的"偷懒依赖"，迫使网络从 S1 雷达纹理中硬推 FPAR。
        # MODIS 只通过 L_cons 和 L_phru 提供知识蒸馏监督。
        bottleneck = self.cross_transformer(f4, plru)

        # 5. Decoder
        pred_hr = self.decoder(bottleneck, [f1, f2, f3])

        if self.training or modis_lr is not None:
            return pred_hr, plru, phru
        else:
            return pred_hr


# ══════════════════════════════════════════════════════════════════════════════
# 7. 三重损失函数
# ══════════════════════════════════════════════════════════════════════════════

class CrossScaleLoss(nn.Module):
    """
    三重约束损失函数:
      L_total = L_cont (纹理重构) + λ1 * L_cons (物理一致) + λ2 * L_temp (时间连续)
    """
    def __init__(self, lambda_cons=0.1, lambda_temp=0.1, valid_threshold=0.05):
        super().__init__()
        self.lambda_cons = lambda_cons
        self.lambda_temp = lambda_temp
        self.threshold = valid_threshold
        self.huber = nn.HuberLoss(reduction='none', delta=0.1)

    def _masked_loss(self, pred, target, mask):
        """计算掩膜后的 Huber Loss"""
        valid_count = mask.sum()
        if valid_count < 2:
            return pred.new_tensor(0.0)
        loss = (self.huber(pred, target) * mask).sum() / valid_count
        return loss

    def _pearson_loss(self, pred, target, mask):
        """计算掩膜后的 Pearson 相关性损失"""
        p = pred[mask > 0]
        t = target[mask > 0]
        if p.numel() < 2:
            return pred.new_tensor(0.0)
        mean_p, std_p = p.mean(), p.std() + 1e-8
        mean_t, std_t = t.mean(), t.std() + 1e-8
        cov = ((p - mean_p) * (t - mean_t)).mean()
        pearson = cov / (std_p * std_t)
        return 1.0 - pearson

    def forward(self, pred_hr, plru, phru, label_hr, modis_lr,
                prev_pred=None, delta_t=None):
        """
        pred_hr:   (B, 1, H, W)     - 最终 10m FPAR 预测
        plru:      (B, 1, 5, 5)     - 聚合生成的低分伪 FPAR
        phru:      (B, 1, H, W)     - 拆解生成的高分伪 FPAR
        label_hr:  (B, 1, H, W)     - S2 真值
        modis_lr:  (B, 1, lr_h, lr_w) - MODIS 真值
        prev_pred: (B, 1, H, W)     - 上一个时间步的预测 (可选)
        delta_t:   (B,)              - 每个样本的 S1-S2 时间差天数 (可选)
        """
        # ── L_cont: 纹理重构损失 (主损失) ─────────────────────────────
        mask_hr = (~torch.isnan(label_hr)).float()
        label_clean = torch.nan_to_num(label_hr, nan=0.0)

        l_cont = self._masked_loss(pred_hr, label_clean, mask_hr)
        # Pearson 权重 1.0（配合中值滤波去噪后不需要太激进）
        l_cont += 1.0 * self._pearson_loss(pred_hr, label_clean, mask_hr)

        # ── L_cons: 物理一致性损失 (PLRU vs MODIS) ────────────────────
        # 将 MODIS 下采样到 PLRU 的尺度
        _, _, ph, pw = plru.shape
        modis_ds = F.interpolate(modis_lr, size=(ph, pw), mode='bilinear', align_corners=False)
        mask_lr = (modis_ds > 0.01).float()
        l_cons = self._masked_loss(plru, modis_ds, mask_lr)

        # ── L_phru: 拆解重建损失 (PHRU vs S2 真值) [新增！] ───────────
        # 迫使 S1 特征 (f4) 必须包含足够的空间引导信息，
        # 才能辅助 MODIS 从 5×5 解压回 256×256 的高分图。
        l_phru = self._masked_loss(phru, label_clean, mask_hr)

        # ── L_temp: 时间连续性损失 (可选) ─────────────────────────────
        l_temp = pred_hr.new_tensor(0.0)
        if prev_pred is not None:
            l_temp = F.mse_loss(pred_hr, prev_pred)

        # ── 总损失 ────────────────────────────────────────────────────
        # 时间衰减权重：时差越大，L_cont 贡献越小（缓和线性衰减 + 0.3 底线）
        if delta_t is not None:
            w = torch.clamp(1.0 - delta_t / 30.0, min=0.3)  # (B,)
            w_mean = w.mean()
            l_cont = l_cont * w_mean

        total = (l_cont
                 + self.lambda_cons * l_cons
                 + 0.5 * l_phru
                 + self.lambda_temp * l_temp)

        return total, {
            'L_cont': l_cont.item(),
            'L_cons': l_cons.item(),
            'L_phru': l_phru.item(),
            'L_temp': l_temp.item(),
            'L_total': total.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossScaleFPARNet(in_channels=8, lr_size=5, patch_size=256).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CrossScaleFPARNet 参数量: {total_params:.2f} M")

    # 模拟输入
    s1 = torch.randn(2, 8, 256, 256).to(device)
    modis = torch.randn(2, 1, 5, 5).to(device)

    # 训练模式
    model.train()
    pred, plru, phru = model(s1, modis)
    print(f"Train - pred: {pred.shape}, plru: {plru.shape}, phru: {phru.shape}")

    # 推理模式
    model.eval()
    with torch.no_grad():
        pred = model(s1)  # 不需要 MODIS
    print(f"Eval  - pred: {pred.shape}")

    # 测试损失
    label = torch.rand(2, 1, 256, 256).to(device)
    loss_fn = CrossScaleLoss()
    model.train()
    pred, plru, phru = model(s1, modis)
    loss, details = loss_fn(pred, plru, phru, label, modis)
    print(f"Loss: {details}")
