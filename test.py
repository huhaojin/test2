# # test.py
# import torch
# import os
# from models.VIFnet import *
# #from models.VIFNet import VIFNet_2025
# #from data_utils import DehazingDataset
# from data_utils.data_utils import *
# from torch.utils.data import DataLoader
# import torchvision.utils as vutils
# from tqdm import tqdm
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataroot', type=str, required=True)
# parser.add_argument('--weights', type=str, default='checkpoints/vifnet_2025_best.pth')
# args = parser.parse_args()
#
# model = VIFNet_2025().cuda()
# model.load_state_dict(torch.load(args.weights))
# model.eval()
#
# loaders_={
# 	'RGB_train': RGB_train_loader,
# 	'RGB_test': RGB_test_loader
# }
# loader_train = loaders_[opt.trainset]
# loader_test = loaders_[opt.testset]
# loader_train = loaders_[opt.trainset]
# loader_test = loaders_[opt.testset]
# #dataset = DehazingDataset(root=args.dataroot, mode='test')
# #loader  = DataLoader(dataset, batch_size=1, shuffle=False)
#
# os.makedirs('pred_imgs_rgbt', exist_ok=True)
#
# with torch.no_grad():
#     for i, (hazy_vis, hazy_ir, name) in enumerate(tqdm(loader_test)):
#         hazy_vis = hazy_vis.cuda()
#         hazy_ir  = hazy_ir.cuda()
#         output = model(hazy_vis, hazy_ir)
#         output = torch.clamp(output, 0, 1)
#         vutils.save_image(output, f'pred_imgs_rgbt/{name[0]}')
#
# print('All done! Results saved in pred_imgs_rgbt/')


import os
import argparse
from option import opt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_utils.data_utils import *
from models.VIFnet import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.mssim import MSSSIM


loaders_ = {
    'RGB_train': RGB_train_loader,
    'RGB_test': RGB_test_loader,
    'RGB_val': RGB_val_loader
}
loader = loaders_[opt.valset]
#rgb, ir, gt = next(iter(loader))


def tensorShow(tensors, titles=['haze']):
        fig=plt.figure()
        for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='rgbt', help='dataset')
parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Test imgs folder')
parser.add_argument('--valset', type=str, default='RGB_val')
opt = parser.parse_args()
dataset = opt.task

output_dir = f'pred_imgs_{dataset}/'
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#model_dir = f'trained_models/vifnet.pk.best.best'
model_dir = f'checkpoints/vifnet_2025_final_epoch035.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
state_dict = torch.load(model_dir, map_location=device)
net = VIFNet_Final()
#net = nn.DataParallel(net)
net.load_state_dict(state_dict)
net.eval()
net = net.to(device)
# print(net)
# gt = Image.open('/home/ym/Downloads/datasets/test/clear/01466D.png')
# gt = tfs.Compose([
#     tfs.ToTensor(),
#     tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
# ])(gt)[None, ::]

# rgb = rgb.to(device)
# ir = ir.to(device)
# gt = gt.to(device)

print(f"开始测试数据集: {opt.valset}")
print(f"数据加载器包含 {len(loader)} 个批次")
# 遍历整个测试集
total_ssim = 0.0
total_psnr = 0.0
num_batches = 0

for batch_idx, (rgb, ir, gt) in enumerate(loader):
    print(f'处理批次 {batch_idx + 1}/{len(loader)}')

    rgb = rgb.to(device)
    ir = ir.to(device)
    gt = gt.to(device)

    with torch.no_grad():
        out = net(rgb, ir)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # 计算指标
        batch_ssim = ssim(out, gt).item()
        batch_psnr = psnr(out, gt)

        total_ssim += batch_ssim
        total_psnr += batch_psnr
        num_batches += 1

        # 保存这个批次的所有输出图片
        for i in range(out.shape[0]):
            ts = torch.squeeze(out[i].clamp(0, 1).cpu())
            # 生成唯一的文件名
            img_idx = batch_idx * loader.batch_size + i
            output_path = os.path.join(output_dir, f'pred_{img_idx:05d}.png')
            vutils.save_image(ts, output_path)

            if i == 0:  # 只打印第一张图片的指标作为参考
                print(f'批次 {batch_idx + 1}, 图片 {img_idx}: ssim={batch_ssim:.4f}, psnr={batch_psnr:.4f}')

# 计算平均指标
if num_batches > 0:
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    print('=' * 50)
    print(f'测试完成! 总共处理了 {num_batches} 个批次')
    print(f'平均 SSIM: {avg_ssim:.4f}')
    print(f'平均 PSNR: {avg_psnr:.4f}')
    print(f'所有输出图片已保存到: {output_dir}')



'''
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.VIFNet import VIFNet
from data_utils import DehazingDataset   # 你原来的那个
from tqdm import tqdm

# === 改这三行就行 ===
weights_path = 'checkpoints/VIFNet_240crop_best.pth'   # 你训练保存的权重
dataroot     = './datasets/AirSim-VID'
batch_size   = 1
# =====================

model = VIFNet().cuda()
model.load_state_dict(torch.load(weights_path))
model.eval()

test_dataset = DehazingDataset(root=dataroot, mode='test')  # 官方 test 集
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

psnr_sum = 0.0
with torch.no_grad():
    for hazy_vis, hazy_ir, clear in tqdm(test_loader, desc='Testing Official Test Set'):
        hazy_vis = hazy_vis.cuda()
        hazy_ir  = hazy_ir.cuda()
        clear    = clear.cuda()
        
        output = model(hazy_vis, hazy_ir)
        output = torch.clamp(output, 0, 1)
        
        mse = F.mse_loss(output, clear)
        psnr = 10 * torch.log10(1.0 / mse)                                                                            
        psnr_sum += psnr.item()

avg_psnr = psnr_sum / len(test_loader)
print(f'\nOfficial AirSim-VID Test PSNR: {avg_psnr:.3f} dB')
print('测试完成！这才是你论文里能写的数字！')
'''