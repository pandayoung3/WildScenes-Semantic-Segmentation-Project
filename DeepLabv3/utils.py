import numpy as np
import torch
from PIL import Image
import pandas as pd

def load_image_paths(image_dir):
    from pathlib import Path
    p = Path(image_dir)
    all_path_list = [str(item) for item in list(p.rglob('*.png'))]
    img_path_list = [item for item in all_path_list if item.split('/')[4] == 'image']
    mask_path_list = [item for item in all_path_list if item.split('/')[4] == 'indexLabel']
    img_path_list.sort()
    mask_path_list.sort()
    return img_path_list, mask_path_list

def prepare_data_dict(img_path_list, mask_path_list):
    data_dict = {'ori_img': img_path_list, "mask_img": mask_path_list}
    return pd.DataFrame(data_dict)