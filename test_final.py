# test_final.py  —— 已修复 Attention 炸显存问题（强制 resize）

import torch
from torch.cuda.amp import autocast
from models.VIFnet import VIFNet
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils
import os

# ========== 修改这里：你的图片路径 ==========
rgb_path = "./test_imgs/603_rgb_foggy_0.5.png"
ir_path  = "./test_imgs/603_ir.png"
# ===========================================

# 输出目录
os.makedirs("./result_final", exist_ok=True)


MAX_SIZE = 512


def resize_max(pil_img, max_size=512):
    w, h = pil_img.size
    scale = max_size / max(w, h)
    if scale < 1:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return pil_img

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载模型
model = VIFNet().to(device).eval()
ckpt_path = "./checkpoints/vifnet_2025_final_epoch035.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
print("✅ 模型加载完成")

# ImageNet 标准化
mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

transform = T.ToTensor()

# 读取图片
rgb_pil = Image.open(rgb_path).convert('RGB')
ir_pil  = Image.open(ir_path).convert('L').convert('RGB')

print("原始分辨率:")
print("RGB:", rgb_pil.size)
print("IR :", ir_pil.size)

# ✅ 缩放到安全尺寸
rgb_pil = resize_max(rgb_pil, MAX_SIZE)
ir_pil  = resize_max(ir_pil, MAX_SIZE)

print("送入模型分辨率:")
print("RGB:", rgb_pil.size)
print("IR :", ir_pil.size)

# 转 tensor
rgb = transform(rgb_pil).unsqueeze(0).to(device)
ir  = transform(ir_pil).unsqueeze(0).to(device)

# 标准化
rgb = (rgb - mean) / std
ir  = (ir - mean) / std

# 推理
with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
    out = model(rgb, ir).clamp(0, 1)

# ✅ 反归一化（让图正常）
out = out * std + mean
out = out.clamp(0,1)

# 保存结果
save_path = "./result_final/603_result.png"
vutils.save_image(out.cpu(), save_path)

print("✅ 成功生成:", save_path)
print("🎉 完事，放心出图，不炸显存")
