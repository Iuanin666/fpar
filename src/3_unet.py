# F:\FPAR_project\src\3_unet.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 初始化权重
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)


class DualStreamUNet(nn.Module):
    """
    双流延迟融合 (Dual-Stream Late Fusion) U-Net。
    主分支处理 S1 (VV, VH) 提取高频空间结构。
    元数据分支处理 Delta_Days 提取时间衰变特征，并在高级语义层 (Bottleneck) 进行晚期融合。
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(DualStreamUNet, self).__init__()
        
        # ── 1. 空间主路分支 (处理 VV, VH, Elevation, Slope, Aspect) ──
        # S1 (2通道) + Terrain (3通道) = 5通道
        self.inc = DoubleConv(5, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # ── 2. Bottleneck 融合层 (集成空间特征 512 + 时间特征 64) ──
        # 空间路径到这里是 512 通道。我们将时间特征注入这里，输出 1024。
        self.bottle_space = DoubleConv(512, 512)
        self.bottle_fuse = DoubleConv(512 + 64, 1024)
        
        # ── 3. 时间序列与物候元数据分支 (处理 Delta_Days, DOY) ──
        self.meta_pool = nn.AdaptiveAvgPool2d(1)
        self.meta_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )

        # ── 4. 上采样解码层 ──
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in [self.up1, self.up2, self.up3, self.up4]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.outc.weight, nonlinearity='sigmoid')
        # 初始化截距，规避 0.5 激活导致的回归停滞
        nn.init.constant_(self.outc.bias, -0.4)

    def forward(self, x):
        # 拆分通道: 
        # Spatial: VV, VH, Elevation, Slope, Aspect (B, 5, H, W)
        # Meta: Delta, DOY (B, 2, H, W)
        x_spatial = x[:, 0:5, :, :]
        x_meta    = x[:, 5:7, :, :]

        # ── 空间分支前向 ──
        x1 = self.inc(x_spatial)       # -> (64, H, W)
        x2 = self.down1(x1)       # -> (128, H/2, W/2)
        x3 = self.down2(x2)       # -> (256, H/4, W/4)
        x4 = self.down3(x3)       # -> (512, H/8, W/8)
        
        b_space = self.bottle_space(nn.MaxPool2d(2)(x4)) # -> (512, H/16, W/16)

        # ── 元数据时间分支前向 ──
        # 将空间维池化掉，得到 (B, 1, 1, 1) -> (B, 1)
        m = self.meta_pool(x_meta).view(x_meta.size(0), -1)
        # 经过 MLP 升维到 64: (B, 64)
        m = self.meta_mlp(m)
        # 将 64 维特征广播 (Broadcast) 到与 b_space 同样的空间维度 (B, 64, H/16, W/16)
        m_broadcast = m.unsqueeze(2).unsqueeze(3).expand(-1, -1, b_space.size(2), b_space.size(3))

        # ── 晚期融合 (Late Fusion) ──
        # (B, 512, H/16, W/16) cat (B, 64, H/16, W/16) -> (B, 576, H/16, W/16)
        fused = torch.cat([b_space, m_broadcast], dim=1)
        x5 = self.bottle_fuse(fused)  # -> (B, 1024, H/16, W/16)

        # ── 解码与跳跃连接 ──
        up_x = self.up1(x5)
        up_x = torch.cat([up_x, x4], dim=1)
        up_x = self.conv1(up_x)

        up_x = self.up2(up_x)
        up_x = torch.cat([up_x, x3], dim=1)
        up_x = self.conv2(up_x)

        up_x = self.up3(up_x)
        up_x = torch.cat([up_x, x2], dim=1)
        up_x = self.conv3(up_x)

        up_x = self.up4(up_x)
        up_x = torch.cat([up_x, x1], dim=1)
        up_x = self.conv4(up_x)

        logits = self.outc(up_x)
        # 使用 Sigmoid 激活函数，硬约束到 0~1 的 FPAR 物理范围
        return torch.sigmoid(logits)