import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp


# ==========================================
# 工具函数：颜色空间转换 (可微分)
# ==========================================

def rgb_to_hsv(image):
    """
    将 RGB 图像转换到 HSV 空间。
    输入 image: [Batch, 3, H, W], 范围 [0, 1]
    输出: s (饱和度), v (亮度) [Batch, 1, H, W]
    """
    image = torch.clamp(image, 0.0, 1.0)
    r, g, b = torch.split(image, 1, dim=1)

    max_rgb, _ = torch.max(image, dim=1, keepdim=True)
    min_rgb, _ = torch.min(image, dim=1, keepdim=True)

    v = max_rgb
    delta = max_rgb - min_rgb
    s = delta / (max_rgb + 1e-8)

    return s, v


def rgb_to_yuv(image):
    """标准的 RGB 转 YUV 公式"""
    r, g, b = torch.split(image, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    return y, u, v


# ==========================================
# 1. 基础与辅助 Loss
# ==========================================

class FFTLoss(nn.Module):
    """频域损失 (呼应论文标题)：强迫网络学习高频振幅，恢复粗糙纹理"""

    def __init__(self):
        super(FFTLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        fft_x = torch.fft.rfft2(x + 1e-8, norm='ortho')
        fft_y = torch.fft.rfft2(y + 1e-8, norm='ortho')
        loss_real = self.l1(fft_x.real, fft_y.real)
        loss_imag = self.l1(fft_x.imag, fft_y.imag)
        return loss_real + loss_imag


class DCPLoss(nn.Module):
    """DCP Loss: 辅助去雾，压暗阴影"""

    def __init__(self, kernel_size=15):
        super(DCPLoss, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        min_rgb, _ = torch.min(x, dim=1, keepdim=True)
        dark_channel = -F.max_pool2d(-min_rgb, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return torch.mean(torch.abs(dark_channel))


# ==========================================
# 2. 细节与结构 Loss
# ==========================================

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # relu1_2 (保纹理)
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:16])  # relu3_3 (保结构)
        for param in self.parameters(): param.requires_grad = False

    def forward(self, x):
        feat12 = self.enc_1(x)
        feat33 = self.enc_2(feat12)
        return feat12, feat33


class EnhancedPerceptualLoss(nn.Module):
    """强化感知损失：结合浅层纹理与中层结构"""

    def __init__(self):
        super(EnhancedPerceptualLoss, self).__init__()
        self.vgg_extractor = VGG16FeatureExtractor()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        feat12_out, feat33_out = self.vgg_extractor(output)
        feat12_tgt, feat33_tgt = self.vgg_extractor(target)
        loss_texture = self.l1(feat12_out, feat12_tgt)
        loss_structure = self.l1(feat33_out, feat33_tgt)
        return 0.5 * loss_texture + 1.0 * loss_structure


class GradientLoss(nn.Module):
    """梯度损失 (Sobel)：终极边缘锐化器，专治画面发软、边缘模糊"""

    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        loss = 0
        # 对 RGB 每个通道分别计算梯度
        for c in range(output.size(1)):
            out_c = output[:, c:c + 1, :, :]
            tgt_c = target[:, c:c + 1, :, :]

            out_grad_x = F.conv2d(out_c, self.weight_x, padding=1)
            out_grad_y = F.conv2d(out_c, self.weight_y, padding=1)
            tgt_grad_x = F.conv2d(tgt_c, self.weight_x, padding=1)
            tgt_grad_y = F.conv2d(tgt_c, self.weight_y, padding=1)

            loss += self.l1(out_grad_x, tgt_grad_x) + self.l1(out_grad_y, tgt_grad_y)
        return loss / output.size(1)


# ==========================================
# 3. 色彩 Loss
# ==========================================

class HSVConeLoss(nn.Module):
    """HSV 锥形损失：专治色彩发灰、不鲜艳"""

    def __init__(self):
        super(HSVConeLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        s_out, v_out = rgb_to_hsv(output)
        s_tgt, v_tgt = rgb_to_hsv(target)
        loss_s = self.l1(s_out, s_tgt)
        loss_v = self.l1(v_out, v_tgt)
        return 2.0 * loss_s + 1.0 * loss_v


# ==========================================
# 4. Total Loss (六边形战士满血版)
# ==========================================

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = EnhancedPerceptualLoss()
        self.hsv_loss = HSVConeLoss()
        self.dcp_loss = DCPLoss()
        self.fft_loss = FFTLoss()  # 新增：频域约束
        self.grad_loss = GradientLoss()  # 新增：梯度锐化

    def forward(self, output, target):
        loss_l1 = self.l1_loss(output, target)
        loss_vgg = self.vgg_loss(output, target)
        loss_hsv = self.hsv_loss(output, target)
        loss_dcp = self.dcp_loss(output)
        loss_fft = self.fft_loss(output, target)
        loss_grad = self.grad_loss(output, target)

        # --- 💎 终极微雕配方 ---
        # 1. 基础阵营 (L1, VGG, HSV, DCP)：保持之前的优秀权重，稳定色彩和轮廓。
        # 2. 锐化阵营 (FFT, Grad)：作为“调味料”加入，权重不能太大，专门雕刻鹅卵石缝隙和树叶边缘。

        total = (1.0 * loss_l1) + \
                (1.5 * loss_vgg) + \
                (10.0 * loss_hsv) + \
                (0.1 * loss_dcp) + \
                (0.1 * loss_fft) + \
                (0.5 * loss_grad)  # 梯度权重 0.5，足够切出锋利的边缘

        return total