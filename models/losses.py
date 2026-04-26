# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:

            vgg = vgg19(weights='IMAGENET1K_V1').features[:35]
        except:
            try:

                vgg = vgg19(pretrained=True).features[:35]
            except:

                vgg = vgg19(weights=None).features[:35]
                vgg.load_state_dict(torch.utils.model_zoo.load_url(
                    'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True))

        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.cuda()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.vgg(x), self.vgg(y))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('k', k.repeat(3,1,1,1))   # ← CPU buffer

    def forward(self, x, y):
        def grad(t):
            # 关键：把卷积核搬到和输入相同的 device 上
            kernel = self.k.to(t.device)
            return F.conv2d(t, kernel, padding=1, groups=3)
        return F.l1_loss(torch.abs(grad(x)), torch.abs(grad(y)))

L1    = nn.L1Loss()
PERC  = PerceptualLoss()
EDGE  = EdgeLoss()

def final_criterion(p, g):
    return L1(p, g) + 0.008 * PERC(p, g) + 0.003 * EDGE(p, g)