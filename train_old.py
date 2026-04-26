# train_vifnet.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.VIFnet import VIFNet
from data_utils.data_utils import *
from models.losses import L1_LOSS, PERC_LOSS, EDGE_LOSS
import torch.nn.functional as F
from tqdm import tqdm
import argparse
loaders_={
	'RGB_train': RGB_train_loader,
	'RGB_test': RGB_test_loader
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./datasets/AirSim-VID')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='vifnet_2025')
    return parser.parse_args()

def criterion(pred, gt):
    l1   = L1_LOSS(pred, gt)
    perc = PERC_LOSS(pred, gt)
    edge = EDGE_LOSS(pred, gt)
    return l1 + 0.10 * perc + 0.06 * edge

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # train_dataset = DehazingDataset(root=args.dataroot, mode='train')
    # train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    model = VIFNet().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    for epoch in range(args.epochs):
        model.train()
        tqdm_bar = tqdm(loader_train)
        for hazy_vis, hazy_ir, clear in tqdm_bar:
            hazy_vis = hazy_vis.cuda()
            hazy_ir  = hazy_ir.cuda()
            clear    = clear.cuda()

            optimizer.zero_grad()
            output = model(hazy_vis, hazy_ir)
            loss   = criterion(output, clear)
            loss.backward()
            optimizer.step()

            tqdm_bar.set_description(f'Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.4f}')

        scheduler.step()

        # 每 10 epoch 保存一次
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'{args.save_path}/{args.name}_epoch{epoch+1}.pth')

        # 简单验证（用训练集最后一批估算 PSNR）






        with torch.no_grad():
            psnr = -10 * torch.log10(((output - clear)**2).mean()).item()

            '''
            def psnr(pred, gt):

                pred = pred.clamp(0, 1).cpu().numpy()
                gt = gt.clamp(0, 1).cpu().detach().numpy()
                imdff = pred - gt
                rmse = math.sqrt(np.mean(imdff ** 2))
                if rmse == 0:
                    return 100
                return 20 * math.log10(1.0 / rmse)
            '''
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), f'{args.save_path}/{args.name}_best.pth')
                print(f'\n=== New best PSNR: {best_psnr:.2f} dB ===\n')

    print('Training finished!')