# quick_psnr_check.py
import torch
import torch.nn.functional as F
from models.VIFnet import VIFNet
from data_utils.data_utils import RGB_test_loader

model = VIFNet().cuda().eval()
model.load_state_dict(torch.load("./checkpoints/vifnet_2025_final_epoch035.pth"))

total_psnr = 0
with torch.no_grad():
    for hazy_vis, hazy_ir, clear in RGB_test_loader:
        hazy_vis = hazy_vis.cuda() / 255.0
        hazy_ir  = hazy_ir.cuda() / 255.0
        clear    = clear.cuda() / 255.0  # 关键！GT 也 /255

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
        hazy_vis = (hazy_vis - mean) / std
        hazy_ir  = (hazy_ir  - mean) / std

        output = model(hazy_vis, hazy_ir).clamp(0, 1)
        mse = F.mse_loss(output, clear)
        psnr = 10 * torch.log10(1 / mse)
        total_psnr += psnr.item()
        break  # 只测第一批

print(f"真实 PSNR: {total_psnr:.3f} dB")