import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """将特征图切分为 Patch 并降维"""
    def __init__(self, in_channels=7, embed_dim=512, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x

class DualStreamTransformer(nn.Module):
    """
    双向流 Transformer 架构 (Bi-directional Concept)
    结合了局部卷积流（CNN Stream）和全局注意力流（Transformer Stream）。
    CNN 流提取局部细节响应（物理边缘），Transformer 流捕获全局上下文关联。
    """
    def __init__(self, in_channels=7, out_channels=1, img_size=256):
        super().__init__()
        
        # 1. 局部特征流 (Local CNN Stream)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.GELU())
        
        # 2. 全局特征流 (Global Transformer Stream)
        embed_dim = 512
        patch_size = 8
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # 使用 PyTorch 原生 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, 
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 3. 混合与还原解码器 (Fusion & Disaggregation Decoder)
        self.up1 = nn.ConvTranspose2d(512 + 512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU()
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.GELU()
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU()
        )
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # ── 局部流计算 ──
        c1 = self.conv1(x)      # [B, 64, H, W]
        c2 = self.conv2(c1)     # [B, 128, H/2, W/2]
        c3 = self.conv3(c2)     # [B, 256, H/4, W/4]
        c4 = self.conv4(c3)     # [B, 512, H/8, W/8]
        
        # ── 全局流计算 ──
        t = self.patch_embed(x)               # [B, N, 512]
        t = t + self.pos_embed
        t = self.transformer(t)               # [B, N, 512]
        
        # 还原 Transformer 输出形状
        t = t.transpose(1, 2).view(B, 512, H//8, W//8) # [B, 512, H/8, W/8]
        
        # ── 特征混合对齐 (Feature Mixing) ──
        fuse = torch.cat([c4, t], dim=1)      # [B, 1024, H/8, W/8]
        
        # ── 逐级还原解码 ──
        d1 = self.up1(fuse)                   # [B, 256, H/4, W/4]
        d1 = torch.cat([d1, c3], dim=1)       # 跳跃连接
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)                     # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, c2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)                     # [B, 64, H, W]
        d3 = torch.cat([d3, c1], dim=1)
        d3 = self.dec3(d3)
        
        out = self.final_conv(d3)
        
        # FPAR 物理极值约束
        return torch.sigmoid(out)

if __name__ == "__main__":
    # 简单测试模型前向传播
    model = DualStreamTransformer().cuda()
    inputs = torch.randn(2, 2, 256, 256).cuda()
    outputs = model(inputs)
    print("Output shape:", outputs.shape)
    print("Params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
