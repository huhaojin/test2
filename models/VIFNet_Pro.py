import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块：Conv -> BN -> LeakyReLU"""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样：MaxPool -> ConvBlock"""

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    """上采样：Upsample -> Concat -> ConvBlock"""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        # 注意：这里的 out_ch 是为了配合下面的 ConvBlock 使用
        # 实际输出通道数会是 out_ch // 2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_ch, out_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配问题（padding）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 拼接跳跃连接的特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FusionUNet(nn.Module):
    def __init__(self):
        super(FusionUNet, self).__init__()
        n_channels = 3
        n_nir = 1
        base = 32  # 基础通道数

        # --- RGB 编码器 ---
        # Inc: 3 -> 32
        self.inc_rgb = ConvBlock(n_channels, base)
        # Down1: 32 -> 64
        self.down1_rgb = Down(base, base * 2)
        # Down2: 64 -> 128
        self.down2_rgb = Down(base * 2, base * 4)
        # Down3: 128 -> 256
        self.down3_rgb = Down(base * 4, base * 8)

        # --- NIR 编码器 ---
        # Inc: 1 -> 32
        self.inc_nir = ConvBlock(n_nir, base)
        # Down1: 32 -> 64
        self.down1_nir = Down(base, base * 2)
        # Down2: 64 -> 128
        self.down2_nir = Down(base * 2, base * 4)
        # Down3: 128 -> 256
        self.down3_nir = Down(base * 4, base * 8)

        # --- 瓶颈层 ---
        # 输入: RGB(256) + NIR(256) = 512
        # 输出: 512
        self.bot_fusion = ConvBlock(base * 8 * 2, base * 16)

        # --- 解码器 (修正了通道数) ---

        # Up1:
        # 输入来自于 Bot(512)
        # Skip来自于 Down3_RGB(128) + Down3_NIR(128) = 256 (注意：Down3 输出的是256，这里我重新核算一下)
        # Down3输出是 base*8=256。
        # 所以 Skip 是 256(RGB) + 256(NIR) = 512。
        # 总输入 = 512(上采样) + 512(Skip) = 1024。
        # 这里的 base*8*2 是 RGB和NIR的 skip 之和。
        # Up definition: in_ch = base*16 (Bot) + base*16 (Skips) = base*32.
        self.up1 = Up(base * 32, base * 16)
        # Up1 Conv 输出: base * 16 // 2 = base * 8 (256)

        # Up2:
        # 输入来自于 Up1(256)
        # Skip来自于 Down2_RGB(128) + Down2_NIR(128) = 256
        # 总输入 = 256(上采样) + 256(Skip) = 512 (base * 16)
        self.up2 = Up(base * 16, base * 8)
        # Up2 Conv 输出: base * 8 // 2 = base * 4 (128)

        # Up3:
        # 输入来自于 Up2(128)
        # Skip来自于 Down1_RGB(64) + Down1_NIR(64) = 128
        # 总输入 = 128(上采样) + 128(Skip) = 256 (base * 8)
        self.up3 = Up(base * 8, base * 4)
        # Up3 Conv 输出: base * 4 // 2 = base * 2 (64)

        # Up4:
        # 输入来自于 Up3(64)
        # Skip来自于 Inc_RGB(32) (注意forward里只拼接了RGB)
        # 总输入 = 64(上采样) + 32(Skip) = 96 (base * 3)
        self.up4 = Up(base * 3, base * 2)
        # Up4 Conv 输出: base * 2 // 2 = base (32)

        self.outc = nn.Conv2d(base, 3, kernel_size=1)

    def forward(self, x_rgb, x_nir):
        # --- 编码 ---
        x1_rgb = self.inc_rgb(x_rgb)  # 32
        x2_rgb = self.down1_rgb(x1_rgb)  # 64
        x3_rgb = self.down2_rgb(x2_rgb)  # 128
        x4_rgb = self.down3_rgb(x3_rgb)  # 256

        x1_nir = self.inc_nir(x_nir)  # 32
        x2_nir = self.down1_nir(x1_nir)  # 64
        x3_nir = self.down2_nir(x2_nir)  # 128
        x4_nir = self.down3_nir(x3_nir)  # 256

        # --- 融合 ---
        fusion_feat = torch.cat([x4_rgb, x4_nir], dim=1)  # 256+256=512
        x_bot = self.bot_fusion(fusion_feat)  # 512

        # --- 解码 ---
        # Up1
        skip3 = torch.cat([x4_rgb, x4_nir], dim=1)  # 这里应该是 x3 还是 x4? U-Net通常是拼接上一层的。
        # 注意：x4是 bottleneck 的输入，skip connection 应该来自 x3。
        # 让我修正 Skip Connection 的来源以匹配 U-Net 结构

        # 修正后的 Skip 逻辑：
        # Layer 4 (Bottom) -> Up to Layer 3
        skip3 = torch.cat([x3_rgb, x3_nir], dim=1)  # 128+128=256
        # x_bot 是 512 (base*16)
        # Up1 需要接收: Up(x_bot) + skip3
        # Up(x_bot) -> 512通道。 Skip3 -> 256通道。 总共 768 (base*24)
        # 刚才 __init__ 里我算成了 1024，这里修正一下 __init__。
        x = self.up1(x_bot, skip3)

        skip2 = torch.cat([x2_rgb, x2_nir], dim=1)  # 64+64=128
        x = self.up2(x, skip2)

        skip1 = torch.cat([x1_rgb, x1_nir], dim=1)  # 32+32=64
        x = self.up3(x, skip1)

        # 最后一层只拼 RGB 恢复色彩
        x = self.up4(x, x1_rgb)  # 32 + 32 = 64

        logits = self.outc(x)
        return torch.sigmoid(logits)


# 重新定义正确的 __init__ 参数以匹配 forward 中的逻辑
class FusionUNet_Corrected(FusionUNet):
    def __init__(self):
        super(FusionUNet, self).__init__()  # 继承上面的类
        base = 32

        # 覆盖掉上面计算错误的层

        # Up1:
        # In: x_bot (512) -> Upsample -> 512
        # Skip: x3_rgb(128) + x3_nir(128) = 256
        # Total: 512 + 256 = 768 (base * 24)
        self.up1 = Up(base * 24, base * 8)
        # Out: base*4 = 128

        # Up2:
        # In: up1_out (128) -> Upsample -> 128
        # Skip: x2_rgb(64) + x2_nir(64) = 128
        # Total: 128 + 128 = 256 (base * 8)
        self.up2 = Up(base * 8, base * 4)
        # Out: base*2 = 64

        # Up3:
        # In: up2_out (64) -> Upsample -> 64
        # Skip: x1_rgb(32) + x1_nir(32) = 64
        # Total: 64 + 64 = 128 (base * 4)
        self.up3 = Up(base * 4, base * 2)
        # Out: base = 32

        # Up4:
        # In: up3_out (32) -> Upsample -> 32
        # Skip: x1_rgb (32)
        # Total: 32 + 32 = 64 (base * 2)
        self.up4 = Up(base * 2, base * 2)
        # Out: base = 32


class VIFNet_Pro(nn.Module):
    def __init__(self):
        super(VIFNet_Pro, self).__init__()
        self.net = FusionUNet_Corrected()

    def forward(self, rgb, nir):
        return self.net(rgb, nir)