import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image
from models.model import VIFNet_Pro  # 保持你的引用
import os

# --- 设置 ---
# 你的路径保持不变
TEST_IMAGE_PATH = 'G:/FusionForImageDehze/foggy_0.5/foggy_0.5/test/haze/692_rgb_foggy_0.5.png'
TEST_NIR_PATH = 'G:/FusionForImageDehze/foggy_0.5/foggy_0.5/test/nir/692_thermal_foggy_0.5.png'
MODEL_PATH = './checkpoints/vifnet_pro_epoch_4150.pth'  # 或者 vifnet_pro_epoch_90.pth


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    model = VIFNet_Pro().to(device)

    if os.path.exists(MODEL_PATH):
        # 增加容错：如果新版 PyTorch 报错 weights_only，这行 try-except 可以自动处理
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except TypeError:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))

        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Error: Model path not found!")
        return

    model.eval()

    # 2. 处理图片
    hazy_img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    nir_img = Image.open(TEST_NIR_PATH).convert('L')

    # --- 关键修改 START ---
    # 获取原始图片的宽和高
    raw_w, raw_h = hazy_img.size
    print(f"Original Image Size: {raw_w} x {raw_h}")

    # 确保红外图尺寸和可见光图完全一致，否则网络拼接会报错
    if nir_img.size != (raw_w, raw_h):
        nir_img = nir_img.resize((raw_w, raw_h), Image.BICUBIC)

    # 这里的 transform 只做转 Tensor，不包含 Resize
    transform = tfs.Compose([tfs.ToTensor()])
    # --- 关键修改 END ---

    hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)
    nir_tensor = transform(nir_img).unsqueeze(0).to(device)

    # 3. 推理
    print("Processing...")
    with torch.no_grad():
        output = model(hazy_tensor, nir_tensor)

    # 4. 保存结果
    output_img = output.squeeze(0).cpu()
    output_pil = tfs.ToPILImage()(output_img)

    save_name = 'result_dehazed_fullsize.png'
    output_pil.save(save_name)

    print(f"Success! Result saved as '{save_name}'")
    print(f"Output Size: {output_pil.size}")


if __name__ == '__main__':
    test()