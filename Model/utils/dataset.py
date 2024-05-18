import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import random_split
import pandas as pd
from PIL import UnidentifiedImageError
import PIL.Image as Image

def load(from_dir: str) -> pd.DataFrame:
    if from_dir[-1] != '/' and from_dir[-1] != '\\':
        from_dir += '/'

    data_dict = {
        'ImgPath': [],
        'FileType': [],
        'Width': [],
        'Height': [],
        'Ratio': [],
        'Mode': [],
        'Bands': [],
        'Transparency': [],
        'Animated': [],
        'Category': [],
        'Interior_Style': []
    }

    for category in os.listdir(from_dir):
        category_path = os.path.join(from_dir, category)
        if os.path.isdir(category_path):
            for style in os.listdir(category_path):
                style_path = os.path.join(category_path, style)
                if os.path.isdir(style_path):
                    for img_name in os.listdir(style_path):
                        img_path = os.path.join(style_path, img_name)
                        try:
                            with Image.open(img_path) as im:
                                data_dict['ImgPath'].append(f'{category}\{style}\{img_name}')
                                data_dict['FileType'].append(img_name.split('.')[-1])
                                data_dict['Width'].append(im.size[0])
                                data_dict['Height'].append(im.size[1])
                                data_dict['Ratio'].append(im.size[0] / im.size[1])
                                data_dict['Mode'].append(im.mode)
                                data_dict['Bands'].append(' '.join(im.getbands()))
                                data_dict['Transparency'].append(
                                    True if 'transparency' in im.info or im.mode in ('RGBA', 'RGBa', 'LA', 'La', 'PA') else False
                                )
                                data_dict['Animated'].append(im.is_animated if hasattr(im, 'is_animated') else False)
                                data_dict['Category'].append(category)
                                data_dict['Interior_Style'].append(style)
                        except (UnidentifiedImageError, PermissionError) as e:
                            print(f"Error processing {img_path}: {e}")

    return pd.DataFrame(data_dict)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters:
        ------------
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        
        Returns:
        ------------
        Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


