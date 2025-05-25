import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop, hflip

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, file_list, image_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.image_transform = image_transform
        self.label_transform = label_transform

        with open(file_list, 'r') as f:
            self.file_names = f.read().splitlines()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, img_name)).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            image_array = np.array(label)
            # label=Image.fromarray(image_array.astype('uint8'))
            label = self.label_transform(label)

            image_array = np.array(label)
        return image, label

class RandomResizeCropFlip:
    def __init__(self, size,is_label):
        self.size = size
        self.is_label =is_label



    def __call__(self, img):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        img_array = np.array(img)
        img = crop(img, i, j, h, w)
        img_array = np.array(img)
        if self.is_label:
            img = img.resize((self.size, self.size), Image.NEAREST)
            img_array = np.array(img)
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)

        # 如果提供了标签，也进行相同的操作
        if np.random.rand() > 0.5:
            img = hflip(img)
            img_array = np.array(img)
        return img

class Resize:
    def __init__(self, size,is_label=True):
        self.size = size
        self.is_label =is_label
    def __call__(self, label):
        if self.is_label:
        #img = img.resize(self.size, Image.BILINEAR)

            label = label.resize(self.size,Image.Resampling.NEAREST)
        else:
            label = label.resize(self.size, Image.Resampling.BILINEAR)

        return label

# 定义验证集的图像和标签的转换
# label_val_transform = transforms.Compose([
#     Resize((512, 512)),
#     label_transform
# ])