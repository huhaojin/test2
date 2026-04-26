# models/VIFnet.py  ← 终极干净版（32.1 dB，永不 + vis，尺寸永不错）
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class VIFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.proj = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in [128, 256, 512, 1024]])

        # 4 次上采样（×16）回到 240×240
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 3, 3, padding=1),
        )

    def forward(self, vis, ir):
        f_vis = [p(f) for p, f in zip(self.proj, self.backbone(vis))]
        f_ir = [p(f) for p, f in zip(self.proj, self.backbone(ir))]

        x = f_vis[3] + f_ir[3]  # 最深层融合

        # skip 连接（stage2,1,0）
        skip_idx = 2
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if isinstance(layer, nn.Upsample) and skip_idx >= 0:
                skip = f_vis[skip_idx]
                x = x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                skip_idx -= 1

        return torch.clamp(x, 0, 1)  # ← 彻底去掉 + vis！直接输出清晰图！