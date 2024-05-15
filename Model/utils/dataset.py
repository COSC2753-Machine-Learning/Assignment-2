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
                                data_dict['ImgPath'].append(f'{category}/{style}/{img_name}')
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


def get_dataloaders_mnist(batch_size, num_workers=0,
                          validation_fraction=None,
                          train_transforms=None,
                          test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 60000)
        train_indices = torch.arange(0, 60000 - num)
        valid_indices = torch.arange(60000 - num, 60000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader


def get_dataloaders_cifar10(batch_size, num_workers=0,
                            validation_fraction=None,
                            train_transforms=None,
                            test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root='data',
                                     train=True,
                                     transform=train_transforms,
                                     download=True)

    valid_dataset = datasets.CIFAR10(root='data',
                                     train=True,
                                     transform=test_transforms)

    test_dataset = datasets.CIFAR10(root='data',
                                    train=False,
                                    transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 50000)
        train_indices = torch.arange(0, 50000 - num)
        valid_indices = torch.arange(50000 - num, 50000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader

def get_dataloaders_local(path, batch_size, train_transform=None, test_transform=None, train_val_test_split=(0.7, 0.2, 0.1), num_workers=0):
    """
    Loads data and splits it into train, validation, and test datasets.
    Expects a single directory with subdirectories for each class.

    Args:
    - path (str): Directory path to the data.
    - batch_size (int): Number of samples per batch.
    - train_transform (callable): Transformations to apply to the training data.
    - test_transform (callable): Transformations to apply to the validation and test data.
    - train_val_test_split (tuple): A tuple indicating the fraction of train, validation, and test datasets.
    - num_workers (int): Number of subprocesses to use for data loading.

    Returns:
    - train_loader, valid_loader, test_loader: Data loaders for the train, validation, and test datasets.
    """
    
    # Load the dataset
    dataset = datasets.ImageFolder(root=path, transform=train_transform)

    # Determine sizes for each split
    total_size = len(dataset)
    train_size = int(train_val_test_split[0] * total_size)
    valid_size = int(train_val_test_split[1] * total_size)
    test_size = total_size - train_size - valid_size

    # Split the dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # Apply test transform to validation and test datasets
    valid_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader