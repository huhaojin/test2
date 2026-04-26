import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 模块 1: FreqEnhancedBlock (GroupNorm版)
# ==========================================
class FreqEnhancedBlock(nn.Module):
    def __init__(self, channels):
        super(FreqEnhancedBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect')
        self.conv_freq = nn.Conv2d(channels * 2, channels * 2, 1)
        self.gn = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_spatial = self.conv1(x)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        real = x_fft.real
        imag = x_fft.imag
        f_cat = torch.cat([real, imag], dim=1)
        f_out = self.conv_freq(f_cat)
        real_out, imag_out = torch.chunk(f_out, 2, dim=1)
        x_fft_out = torch.complex(real_out, imag_out)
        x_freq_rec = torch.fft.irfft2(x_fft_out, s=x.shape[-2:], norm='ortho')
        return self.relu(self.gn(x_spatial + x_freq_rec))


# ==========================================
# 模块 2: SFTLayer (保持不变)
# ==========================================
class SFTLayer(nn.Module):
    def __init__(self, channels):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(channels, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, channels, 1)
        self.SFT_shift_conv0 = nn.Conv2d(channels, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, channels, 1)
        # SFT的初始化还是保留比较好，有助于初期稳定
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.SFT_scale_conv1.weight, 0)
        nn.init.constant_(self.SFT_scale_conv1.bias, 0)
        nn.init.constant_(self.SFT_shift_conv1.weight, 0)
        nn.init.constant_(self.SFT_shift_conv1.bias, 0)

    def forward(self, x_rgb, x_nir_feat):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x_nir_feat), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x_nir_feat), 0.1, inplace=True))
        return x_rgb * (scale + 1) + shift


# ==========================================
# 模块 3: ResBlock_SFT (GroupNorm版)
# ==========================================
class ResBlock_SFT(nn.Module):
    def __init__(self, channels, dilation=1):
        super(ResBlock_SFT, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation,
                               bias=False, padding_mode='reflect')
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.sft1 = SFTLayer(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation,
                               bias=False, padding_mode='reflect')
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.sft2 = SFTLayer(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, nir_feat):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.sft1(out, nir_feat)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.sft2(out, nir_feat)
        out += residual
        return self.relu(out)


# ==========================================
# 主模型: VIFNet_Pro (Direct Prediction版)
# ==========================================
class VIFNet_Pro(nn.Module):
    def __init__(self):
        super(VIFNet_Pro, self).__init__()
        base_channel = 64

        # 1. RGB 入口
        self.rgb_entry = nn.Sequential(
            nn.Conv2d(3, base_channel, 3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups=8, num_channels=base_channel),
            nn.ReLU(inplace=True),
            FreqEnhancedBlock(base_channel)
        )

        # 2. NIR 入口
        self.nir_entry = nn.Sequential(
            nn.Conv2d(1, base_channel, 3, padding=1, padding_mode='reflect'),
            nn.GroupNorm(num_groups=8, num_channels=base_channel),
            nn.ReLU(inplace=True),
            FreqEnhancedBlock(base_channel)
        )

        # 3. 主干
        self.sft_blocks = nn.ModuleList([
            ResBlock_SFT(base_channel, dilation=1),
            ResBlock_SFT(base_channel, dilation=2),
            ResBlock_SFT(base_channel, dilation=3),
            ResBlock_SFT(base_channel, dilation=5),
            ResBlock_SFT(base_channel, dilation=2),
            ResBlock_SFT(base_channel, dilation=1),
            ResBlock_SFT(base_channel, dilation=1),
            ResBlock_SFT(base_channel, dilation=1)
        ])

        # 4. 重建层 (取消了 Zero Init，改为默认随机初始化)
        self.tail = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel // 2, 3, 3, padding=1, padding_mode='reflect')
        )
        # 【注意】这里删除了之前的 nn.init.constant_ 零初始化代码
        # 让它随机初始化，逼迫模型必须通过学习才能输出正常的图像

    def forward(self, x_rgb, x_nir):
        f_rgb = self.rgb_entry(x_rgb)
        f_nir = self.nir_entry(x_nir)

        x = f_rgb
        for block in self.sft_blocks:
            x = block(x, f_nir)

        # 【核心修改】直接预测 (Direct Prediction)
        # 以前是: out = x_rgb + self.tail(x)  (残差学习)
        # 现在是: out = self.tail(x)          (直接学习)
        # 解释：去掉 x_rgb 的跳跃连接，模型被迫完全重绘图像，无法再“偷懒”输出原图。
        out = self.tail(x)

        return torch.clamp(out, 0, 1)