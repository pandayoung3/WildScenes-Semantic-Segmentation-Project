import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class WSdataset(Dataset):
    def __init__(self, df_data):
        self.ori_img_path_list = list(df_data['ori_img'])
        self.mask_img_path_list = list(df_data['mask_img'])

    def __len__(self):
        return len(self.ori_img_path_list)

    def __getitem__(self, idx):
        image_path = self.ori_img_path_list[idx]
        mask_path = self.mask_img_path_list[idx]
        image = np.array(Image.open(image_path).resize((256, 256)).convert("RGB"))
        mask = np.array(Image.open(mask_path).resize((256, 256))).astype('int32')

        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image).float()
        mask = torch.tensor(mask).long()

        return image, mask