import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from cv2 import transform
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
import torchvision.utils as vutils
from auto_augment import AutoAugment
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import time

import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from PIL import Image

# 定义数据增强操作
transform = transforms.Compose([
    AutoAugment(),
    transforms.RandomAffine(30),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),  # 随机旋转20度以内
    transforms.RandomResizedCrop(64),  # 随机裁剪为224x224大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

])

# 加载图片
image = Image.open('angry2.jpg')

# 应用数据增强操作
augmented_images = []
for i in range(20):  # 生成10张增强后的图片
    augmented_image = transform(image)
    augmented_images.append(augmented_image)

# 创建数据加载器
dataloader = DataLoader(augmented_images, batch_size=1, shuffle=True)

# 遍历数据加载器
# for i, batch in enumerate(dataloader):
#     augmented_image = batch[0].numpy().transpose((1, 2, 0))
#     augmented_image = (augmented_image * 255).astype('uint8')
#
#     # 保存增强后的图片
#     save_image('output{}.jpg'.format(i), augmented_image)
for i, batch in enumerate(dataloader):
    augmented_image = batch[0]
    # 保存增强后的图片
    vutils.save_image(augmented_image, 'output/output{}.jpg'.format(i))
