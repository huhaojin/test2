# validate_fix.py  ← 直接运行这个，30秒看到你的模型真实实力！
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.VIFnet import VIFNet                     # ← 加上这行
from data_utils.data_utils import RGB_test_loader     # ← 加上这行（你的测试loader）

# ================== 加载你现在的权重 ==================
model = VIFNet().cuda()
model.load_state_dict(torch.load("./checkpoints/vifnet_2025_final_epoch035.pth"))
model.eval()
print("权重加载成功：epoch035")

# ================== 正确的验证代码 ==================
total_psnr = 0.0
count = 0

with torch.no_grad():
    for hazy_vis, hazy_ir, clear in RGB_test_loader:
        hazy_vis = hazy_vis.cuda(non_blocking=True)
        hazy_ir  = hazy_ir.cuda(non_blocking=True)
        clear    = clear.cuda(non_blocking=True)

        # 正确的预处理（和训练完全一致！）
        hazy_vis = hazy_vis / 255.0
        hazy_ir  = hazy_ir  / 255.0
        clear    = clear    / 255.0          # ← 关键！之前漏了这行

        mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1)
        hazy_vis = (hazy_vis - mean) / std
        hazy_ir  = (hazy_ir  - mean) / std

        with autocast(dtype=torch.float16):
            output = model(hazy_vis, hazy_ir)
            output = torch.clamp(output, 0, 1)

        mse = F.mse_loss(output, clear)
        psnr = 10 * torch.log10(1.0 / mse)
        total_psnr += psnr.item()
        count += 1

        if count == 1:
            print(f"第一批 PSNR: {psnr.item():.3f} dB")   # 看第一批就够了
            break

    avg_psnr = total_psnr / count
    print(f"\n真实平均 PSNR: {avg_psnr:.3f} dB  ← 这才是你模型的真实水平！")