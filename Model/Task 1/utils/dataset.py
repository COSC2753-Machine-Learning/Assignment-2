import pandas as pd
import os
import imagehash
from PIL import Image

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
        'Class': []
    }

    for flower_class in os.listdir(from_dir):
        for ImgPath in os.listdir(f'{from_dir}{flower_class}'):
            with Image.open(f'{from_dir}{flower_class}/{ImgPath}') as im:
                data_dict['ImgPath'].append(f'{flower_class}/{ImgPath}')
                data_dict['FileType'].append(ImgPath.split('.')[-1])
                data_dict['Width'].append(im.size[0])
                data_dict['Height'].append(im.size[1])
                data_dict['Ratio'].append(im.size[0] / im.size[1])
                data_dict['Mode'].append(im.mode)
                data_dict['Bands'].append(' '.join(im.getbands()))
                data_dict['Transparency'].append(
                    True
                    if (im.mode in ('RGBA', 'RGBa', 'LA', 'La', 'PA')) or (im.mode == 'P' and 'transparency' in im.info)
                    else False
                )
                data_dict['Animated'].append(im.is_animated if hasattr(im, 'is_animated') else False)
                data_dict['Class'].append(flower_class)

    return pd.DataFrame(data_dict)