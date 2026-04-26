import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math

# 假设你的模型文件名为 models/model.py
from models.model import VIFNet_Pro
from models.loss_pro import TotalLoss
from data_utils.data_loader import VIFDataset

# --- 参数配置 ---
# 显存够大(3090/4090)设为 4 或 8，显存小(8G/12G)务必设为 2
CROP_SIZE = 512

# 这样梯度更稳，训练更快
BATCH_SIZE = 1

# 总 Epochs
EPOCHS = 4500

# 基础学习率 (重置后的起点)
LR = 1e-5

# 你的真实路径
DATA_PATH = r'G:/FusionForImageDehze/foggy_0.5/foggy_0.5/train'
SAVE_DIR = './checkpoints'

# 断点续训：设置为你当前的 Epoch (比如 240)
# 这会加载 epoch_240.pth，然后强制重置学习率继续跑
RESUME_EPOCH = 3980


def calculate_psnr(img1, img2):
    """
    计算两个 Tensor 图片之间的 PSNR (峰值信噪比)
    img1, img2: [B, 3, H, W], 范围 [0, 1]
    """
    with torch.no_grad():
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()


def train():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 1. 准备数据
    print(f"Loading Data from {DATA_PATH}...")
    try:
        train_set = VIFDataset(DATA_PATH, train=True, size=CROP_SIZE)
        # Windows 下 num_workers=0 比较稳
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. 准备模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VIFNet_Pro().to(device)
    criterion = TotalLoss().to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 调整学习率策略 (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # =================================================================
    # [修复点] 必须先给 start_epoch 一个默认值
    # =================================================================
    start_epoch = 0

    checkpoint_path = os.path.join(SAVE_DIR, f'vifnet_pro_epoch_{RESUME_EPOCH}.pth')

    # 只有当 RESUME_EPOCH > 0 且文件存在时才加载
    if RESUME_EPOCH > 0:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")

            # 1. 加载权重 (strict=False 忽略不匹配层，比如归一化层的微小差异)
            try:
                # 注意：如果你的pth文件直接保存的是 state_dict (看你之前的代码是这样的)，直接 load 即可
                # 如果是保存的 {'model': ..., 'epoch': ...} 字典，则需要 checkpoint['model']
                # 根据你提供的 save 代码 `torch.save(model.state_dict(), ...)`，这里直接 load 是对的。
                state_dict = torch.load(checkpoint_path)
                model.load_state_dict(state_dict, strict=False)
                print("✅ Successfully loaded weights! (Partial load if logic changed)")

                start_epoch = RESUME_EPOCH

                # =========================================================
                # 【核心操作】 激进微调策略：强制重置学习率
                # =========================================================
                print(f"⚡️ [Aggressive Strategy] Force Resetting Learning Rate to {LR}...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR

                # 重置 Scheduler，让它以为是从头开始衰减 (或者从当前epoch开始新的衰减周期)
                # 这里为了简单，让它继续按 Cosine 走，但起点已经是 1e-4 了
                # 如果想让它 decay 慢一点，可以重新定义 scheduler
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7,
                                                                 last_epoch=start_epoch)

                print(f"Resuming training from Epoch {start_epoch} with RESET LR...")

            except Exception as e:
                print(f"Weight loading warning: {e}")
                # 如果加载失败，建议检查路径，或者暂时设 RESUME_EPOCH=0 从头跑验证代码
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found! Starting from scratch.")
            start_epoch = 0
    else:
        print("Starting from scratch...")

    print(f"Start Training from Epoch {start_epoch} to {EPOCHS}...")

    # 3. 训练循环
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0  # 初始化 PSNR 累加器

        # 遍历数据
        for i, (hazy, clear, nir) in enumerate(train_loader):
            hazy = hazy.to(device)
            clear = clear.to(device)
            nir = nir.to(device)

            # Forward
            output = model(hazy, nir)

            # Loss 计算
            loss = criterion(output, clear)

            # PSNR 计算 (仅用于显示，不参与反向传播)
            current_psnr = calculate_psnr(output, clear)
            epoch_psnr += current_psnr

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸，这对 GroupNorm + 激进 Loss 很有必要)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, PSNR: {current_psnr:.2f} dB")

        scheduler.step()

        # 打印当前学习率 (确认重置是否生效)
        current_lr = optimizer.param_groups[0]['lr']

        # 计算本轮平均指标
        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)

        # 打印日志
        print(
            f"===> Epoch {epoch + 1} Complete: Avg. Loss: {avg_loss:.4f} | Avg. PSNR: {avg_psnr:.2f} dB | LR: {current_lr:.2e}")

        # 每 10 个 Epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f'vifnet_pro_epoch_{epoch + 1}.pth')
            # 只保存权重，方便加载
            torch.save(model.state_dict(), save_path)
            print("-" * 50)
            print(f"✅ Checkpoint saved: {save_path}")
            print(f"📊 Milestone Metrics - Epoch {epoch + 1}: PSNR = {avg_psnr:.2f} dB")
            print("-" * 50)

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vifnet_pro_final.pth'))
    print("Training Finished!")


if __name__ == '__main__':
    train()