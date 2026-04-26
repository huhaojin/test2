import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------------------------------
# 1. 辅助函数：计算某种加权平均（论文中常见的融合系数）
# -------------------------------------------------
def f(x, y):
    # 原实现（被注释掉）：1/2 * (1-x)*(1-y) + x*y
    # return 1 / 2 * (1 - x) * (1 - y) + x * y
    # 当前实现：(1-x)*(1-y) + 1/2 * x*y   （等价于对 (x,y) 的线性插值）
    return (1 - x) * (1 - y) + 1 / 2 * x * y


# -------------------------------------------------
# 2. 边缘检测算子（Sobel、Robert），转为 PyTorch Tensor
# -------------------------------------------------
Sobel = np.array([[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]])               # 水平/垂直梯度核
Robert = np.array([[0, 0],
                   [-1, 1]])                 # Robert 交叉差分核（这里只用了最简版）
Sobel = torch.Tensor(Sobel)                     # (3,3)
Robert = torch.Tensor(Robert)                   # (2,2)


# -------------------------------------------------
# 3. 二值化模块：把特征图归一化到 [0,1]
# -------------------------------------------------
class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)      # 全局平均池化 → (B,C,1,1)

    def forward(self, x):
        y = self.avg(x)                         # 每个通道的全局均值
        x = torch.sign(x - y)                   # 大于均值 → +1，小于均值 → -1
        out = (x + 1) / 2                        # 映射到 {0,1}
        return out


# -------------------------------------------------
# 4. 统一的卷积封装（保持特征图尺寸不变）
# -------------------------------------------------
def Conv(in_channels, out_channels, kernel_size, stride=1, bias=False):
    """
    标准 Conv2d，padding = kernel_size//2 → 输出尺寸 = 输入尺寸（stride=1 时）
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


# -------------------------------------------------
# 5. 注释掉的 SE-Layer（Squeeze-Excitation）和 RCAB
# -------------------------------------------------
# class SELayer(nn.Module): ...
# class RCAB(nn.Module): ...


# -------------------------------------------------
# 6. PA-Layer（Pixel Attention）——通道维度的注意力
# -------------------------------------------------
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )                                      # 输出 (B,1,H,W) 的空间注意力图

    def forward(self, x):
        y = self.pa(x)                         # 生成注意力权重
        return x * y                           # 逐像素加权


# -------------------------------------------------
# 7. CA-Layer（Channel Attention）——全局通道注意力
# -------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # (B,C,1,1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )                                      # 输出 (B,C,1,1) 的通道权重

    def forward(self, x):
        y = self.avg_pool(x)                   # 全局平均
        y = self.ca(y)                         # 通道注意力向量
        return x * y                           # 逐通道加权


# -------------------------------------------------
# 8. CPAB（Conv + PReLU + Conv + CA + PA + Residual）
# -------------------------------------------------
class CPAB(nn.Module):
    """
    核心特征提取块：两层卷积 + 残差 + 通道注意力 + 像素注意力
    """
    def __init__(self, dim, kernel_size, bias):
        super(CPAB, self).__init__()
        self.conv1 = Conv(dim, dim, kernel_size, bias=bias)
        self.act1  = nn.PReLU()                 # 可学习参数的 ReLU
        self.conv2 = Conv(dim, dim, kernel_size, bias=bias)
        self.calayer = CALayer(dim)            # 通道注意力
        self.palayer = PALayer(dim)            # 像素注意力

    def forward(self, x):
        res = self.act1(self.conv1(x))         # 第一层卷积 + 激活
        res = res + x                          # 第一次残差
        res = self.conv2(res)                  # 第二层卷积
        res = self.calayer(res)                # 通道注意力
        res = self.palayer(res)                # 像素注意力
        res += x                               # 第二次残差（整体残差）
        return res


# -------------------------------------------------
# 9. 输出头：特征 → RGB（或指定通道数），可选残差连接
# -------------------------------------------------
class Output(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, output_channel=3, residual=True):
        super(Output, self).__init__()
        self.conv = Conv(n_feat, output_channel, kernel_size, bias=bias)
        self.residual = residual               # 是否把输入图像加到输出上

    def forward(self, x, x_img):
        """
        x      : 最后一层特征图 (B, n_feat, H, W)
        x_img  : 原始低分辨率图像 (B, output_channel, H, W)
        """
        x = self.conv(x)
        if self.residual:
            x += x_img                         # 残差学习（输出 = 特征 + 原图）
        return x


# -------------------------------------------------
# 10. Encoder（三层下采样 + CPAB + 可选特征注意力）
# -------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, atten):
        super(Encoder, self).__init__()
        self.atten = atten                     # 是否在推理阶段使用外部特征注意力

        # 三层 CPAB（通道数逐步翻倍）
        self.encoder_level1 = CPAB(n_feat, kernel_size, bias=bias)
        self.encoder_level2 = CPAB(n_feat*2, kernel_size, bias=bias)
        self.encoder_level3 = CPAB(n_feat*4, kernel_size, bias=bias)

        # 下采样（0.5×）
        self.down12 = DownSample(n_feat, n_feat*2)
        self.down23 = DownSample(n_feat*2, n_feat*4)

        # 特征注意力（仅在 atten=True 时使用）
        if self.atten:
            self.atten_conv1 = Conv(n_feat, n_feat, 1, bias=bias)
            self.atten_conv2 = Conv(n_feat*2, n_feat*2, 1, bias=bias)
            self.atten_conv3 = Conv(n_feat*4, n_feat*4, 1, bias=bias)

    def forward(self, x, encoder_outs=None):
        """
        两种模式：
        1. encoder_outs is None → 正常编码，返回 [enc1, enc2, enc3]
        2. encoder_outs is not None → 融合外部特征（常用于多模态/双分支网络）
        """
        if encoder_outs is None:
            enc1 = self.encoder_level1(x)
            x    = self.down12(enc1)
            enc2 = self.encoder_level2(x)
            x    = self.down23(enc2)
            enc3 = self.encoder_level3(x)
            return [enc1, enc2, enc3]
        else:
            # 融合外部特征（1×1 卷积对齐通道）
            enc1 = self.encoder_level1(x)
            enc1_fuse_nir = enc1 + self.atten_conv1(encoder_outs[0])
            x = self.down12(enc1_fuse_nir)

            enc2 = self.encoder_level2(x)
            enc2_fuse_nir = enc2 + self.atten_conv2(encoder_outs[1])
            x = self.down23(enc2_fuse_nir)

            enc3 = self.encoder_level3(x)
            enc3_fuse_nir = enc3 + self.atten_conv3(encoder_outs[2])
            return [enc1_fuse_nir, enc2_fuse_nir, enc3_fuse_nir]


# -------------------------------------------------
# 11. Decoder（三层上采样 + CPAB + 跳连残差）
# -------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, residual=True):
        super(Decoder, self).__init__()
        self.residual = residual

        self.decoder_level1 = CPAB(n_feat, kernel_size, bias=bias)
        self.decoder_level2 = CPAB(n_feat*2, kernel_size, bias=bias)
        self.decoder_level3 = CPAB(n_feat*4, kernel_size, bias=bias)

        # 跳连卷积（对齐通道）
        self.skip_conv_1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.skip_conv_2 = Conv(n_feat*2, n_feat*2, kernel_size, bias=bias)

        # 上采样（2×）
        self.up21 = UpSample(n_feat*2, n_feat)
        self.up32 = UpSample(n_feat*4, n_feat*2)

    def forward(self, outs):
        """
        outs = [enc1, enc2, enc3]（Encoder 的输出）
        """
        enc1, enc2, enc3 = outs

        dec3 = self.decoder_level3(enc3)               # 最深层处理

        x = self.up32(dec3)                            # 上采样到 1/2 尺度
        if self.residual:
            x += self.skip_conv_2(enc2)                # 跳连残差
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2)                            # 上采样到原尺度
        if self.residual:
            x += self.skip_conv_1(enc1)                # 跳连残差
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]                       # 返回三层解码特征（可用于多任务）


# -------------------------------------------------
# 12. DownSample（0.5×）——先 bilinear 再 1×1 Conv
# -------------------------------------------------
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(DownSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


# -------------------------------------------------
# 13. UpSample（2×）——先 bilinear 再 1×1 Conv
# -------------------------------------------------
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UpSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


# -------------------------------------------------
# 14. Edge 检测模块（基于 Sobel 算子）
# -------------------------------------------------
class Edge(nn.Module):
    """
    输入任意通道数特征图，输出同尺寸的梯度幅度图（每个通道独立计算）
    """
    def __init__(self, channel, kernel='sobel'):
        super(Edge, self).__init__()
        self.channel = channel
        self.kernel = kernel

        # Sobel 核扩展到 (C,1,3,3)
        self.kernel_x = Sobel.repeat(channel, 1, 1, 1)      # 水平梯度
        self.kernel_y = self.kernel_x.permute(0, 1, 3, 2)   # 垂直梯度（转置）
        self.kernel_x = nn.Parameter(self.kernel_x, requires_grad=False)
        self.kernel_y = nn.Parameter(self.kernel_y, requires_grad=False)

    def forward(self, current):
        """
        current : (B, C, H, W)
        """
        current = F.pad(current, (1, 1, 1, 1), mode='reflect')   # 防止边界效应
        gradient_x = torch.abs(F.conv2d(current,
                                        weight=self.kernel_x,
                                        groups=self.channel,
                                        padding=0))
        gradient_y = torch.abs(F.conv2d(current,
                                        weight=self.kernel_y,
                                        groups=self.channel,
                                        padding=0))
        out = gradient_x + gradient_y                           # 梯度幅度
        return out