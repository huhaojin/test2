import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
import torch


class VIFDataset(data.Dataset):
    def __init__(self, path, train=True, size=256):
        self.train = train
        self.size = size

        # 1. 定义路径
        self.root_hazy = os.path.join(path, 'haze')
        self.root_clear = os.path.join(path, 'clear')
        self.root_nir = os.path.join(path, 'nir')

        # 2. 获取文件名
        self.hazy_imgs = sorted(os.listdir(self.root_hazy))
        self.clear_imgs = sorted(os.listdir(self.root_clear))
        self.nir_imgs = sorted(os.listdir(self.root_nir))

        print(f"Dataset loaded: {len(self.hazy_imgs)} images.")

    def __getitem__(self, index):
        # 1. 读取图片
        hazy_path = os.path.join(self.root_hazy, self.hazy_imgs[index])
        clear_path = os.path.join(self.root_clear, self.clear_imgs[index])
        nir_path = os.path.join(self.root_nir, self.nir_imgs[index])

        try:
            hazy = Image.open(hazy_path).convert('RGB')
            clear = Image.open(clear_path).convert('RGB')
            nir = Image.open(nir_path).convert('L')  # 【关键】NIR 必须转为 'L' (灰度1通道)
        except Exception as e:
            print(f"Error loading image index {index}: {e}. Skipping...")
            return self.__getitem__(random.randint(0, len(self.hazy_imgs) - 1))

        # 2. 数据增强
        if self.train:
            # 安全检查：防止图片尺寸小于裁剪尺寸导致报错
            w, h = hazy.size
            if w < self.size or h < self.size:
                hazy = tfs.Resize((self.size, self.size))(hazy)
                clear = tfs.Resize((self.size, self.size))(clear)
                nir = tfs.Resize((self.size, self.size))(nir)

            # 随机裁剪
            i, j, h, w = tfs.RandomCrop.get_params(hazy, output_size=(self.size, self.size))
            hazy = tfs.functional.crop(hazy, i, j, h, w)
            clear = tfs.functional.crop(clear, i, j, h, w)
            nir = tfs.functional.crop(nir, i, j, h, w)

            # 随机水平翻转
            if random.random() > 0.5:
                hazy = tfs.functional.hflip(hazy)
                clear = tfs.functional.hflip(clear)
                nir = tfs.functional.hflip(nir)

        # 3. 转换为 Tensor
        hazy = tfs.ToTensor()(hazy)
        clear = tfs.ToTensor()(clear)
        nir = tfs.ToTensor()(nir)

        # 【核心修复点】就在这里！
        # 必须是 (hazy, clear, nir) 的顺序
        # 这样 train 代码里的 clear 变量才能拿到 clear 图，nir 变量才能拿到 nir 图
        return hazy, clear, nir

    def __len__(self):
        return len(self.hazy_imgs)