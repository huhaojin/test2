# # model/VIFNet.py
#
# import sys
#
# # sys.path.append(r"D:\model_test\InternImage\segmentation")
# # sys.path.append(r"D:\model_test\InternImage\segmentation\ops_dcnv3")
# # sys.path.append(r"D:\model_test\InternImage\segmentation\mmseg_custom\models\backbones")
# sys.path.append(r"D:\model_test\InternImage_Pytorch")
# from modules.internimage import InternImage
# #from intern_image import InternImage
# sys.path.append(r"D:\model_test\VMamba")
# from vmamba import VSSBlock
# #from intern_image import InternImage
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50
# from torchvision.models.feature_extraction import create_feature_extractor
#from ops_dcnv3 import DCNv3
import sys
import timm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, r"G:\FusionForImageDehze\model_test\InternImage_Pytorch")
sys.path.insert(0, r"G:\FusionForImageDehze\model_test\VMamba")

import modules
print(">>> modules 来自：", modules.__file__)

from modules.internimage import InternImage
from vmamba import VSSBlock

# class InternEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = InternImage(
#             core_op='DCNv3',
#             channels=128,
#             depths=[5, 5, 23, 5],
#             groups=[8, 16, 32, 64],
#             offset_scale=1.0,
#             mlp_ratio=4.,
#             post_norm=True,
#             norm_layer='LN',
#             layer_scale=None,
#             drop_path=0.4,
#         )
#         self.proj = nn.ModuleList([
#             nn.Conv2d(c, 256, kernel_size=1, bias=False)
#             for c in [128, 256, 512, 1024]
#         ])
#
#     def forward(self, x):
#         feats = self.backbone(x)          # List[4]: C=[128,256,512,1024]
#         return [self.proj[i](feats[i]) for i in range(4)]



# models/VIFnet.py  ← 只改 ConvNeXtEncoder 这段，其他不动
class ConvNeXtEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        # 自动获取通道数
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy)
        channels = [f.shape[1] for f in feats]
        print("ConvNeXt channels:", channels)  # [128, 256, 512, 1024]
        self.proj = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in channels])

    def forward(self, x):
        feats = self.backbone(x)
        return [proj(f) for proj, f in zip(self.proj, feats)]


class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 投影（可见光 Q，红外 K/V）
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, vis_feat, ir_feat):
        B, C, H, W = vis_feat.shape
        # 展平为序列 [B, HW, C]
        vis_flat = vis_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        ir_flat = ir_feat.flatten(2).transpose(1, 2)  # [B, HW, C]

        # 生成 Q (vis), K/V (ir)
        Q = self.q_proj(vis_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(ir_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(ir_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ V).transpose(1, 2).reshape(B, -1, C)

        # 输出投影 + 残差
        x = self.out_proj(x)
        x = self.norm(x + vis_flat)  # 残差连接

        # 恢复 4D
        return x.transpose(1, 2).reshape(B, C, H, W)


class PureAttentionFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attns = nn.ModuleList([
            CrossModalAttention(dim, num_heads=8) for _ in range(4)
        ])
        self.gate = nn.Conv2d(dim * 2, 2, 1)

    def forward(self, vis_feats, ir_feats):
        fused = []
        for i, (v, i_) in enumerate(zip(vis_feats, ir_feats)):
            # 注意力融合
            attended = self.attns[i](v, i_)
            # 简单 concat + gate
            x = torch.cat([attended, i_], dim=1)
            w = torch.sigmoid(self.gate(x))
            fused.append(w[:, 0:1] * v + w[:, 1:2] * i_)
        return fused


class VIFNet_Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvNeXtEncoder()
        self.fusion = PureAttentionFusion()

        dec = []
        for i in range(4):
            dec.append(nn.Conv2d(256, 256, 3, padding=1))
            dec.append(nn.ReLU(inplace=True))
            if i < 3:
                dec.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        dec.append(nn.Conv2d(256, 3, 3, padding=1))
        self.decoder = nn.Sequential(*dec)

    def forward(self, vis, ir):
        B, C, H, W = vis.shape
        f_vis = self.encoder(vis)
        f_ir = self.encoder(ir)
        f_fused = self.fusion(f_vis, f_ir)

        x = f_fused[3]  # 最深层 H/32
        skip_idx = 2  # 2→1→0

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                skip_feat = f_fused[skip_idx]
                if x.shape[2:] != skip_feat.shape[2:]:
                    skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip_feat
                skip_idx -= 1

        # 关键：最后再 ×4 上采样到原图大小
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return torch.clamp(x + vis, 0, 1)


VIFNet = VIFNet_Final  # 兼容
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VIFNet_Final().to(device)

    x_vis = torch.randn(1, 3, 128, 128, device=device)
    x_ir  = torch.randn(1, 3, 128, 128, device=device)

    y = model(x_vis, x_ir)
    print(y.shape)
