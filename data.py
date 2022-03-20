import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=[-30.0, 30.0], translate=[0.0, 0.5], scale=[0.7, 1.3]),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])


def get_dataset_and_dataloader(cifar_dir='cifar'):
    train_data = datasets.CIFAR100(
        cifar_dir,
        train=True,
        transform=transform_aug
    )

    clean_test_data = datasets.CIFAR100(
        cifar_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    # define dataset
    train_ratio = 0.2
    train_len = int(train_ratio * len(clean_test_data))
    test_len = len(clean_test_data) - train_len
    partial_train_data, partial_test_data = torch.utils.data.random_split(clean_test_data, [train_len, test_len])

    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    full_test_loader = DataLoader(
        clean_test_data,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    return train_data, clean_test_data, train_loader, full_test_loader


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_corrupt_data(corruption, cifar_corrupt_dir='cifar_corrupt/CIFAR-100-C'):
    test_data = datasets.CIFAR100(
        'cifar',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    test_data.data = np.load(os.path.join(cifar_corrupt_dir, corruption + '.npy'))
    test_data.targets = torch.LongTensor(np.load(os.path.join(cifar_corrupt_dir, 'labels.npy')))
    return test_data


def get_corrupt_loader(corruption, batch_size, cifar_corrupt_dir='cifar_corrupt/CIFAR-100-C'):
    test_data = datasets.CIFAR100(
        'cifar',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    test_data.data = np.load(os.path.join(cifar_corrupt_dir, corruption + '.npy'))
    test_data.targets = torch.LongTensor(np.load(os.path.join(cifar_corrupt_dir, 'labels.npy')))
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader


class CycleLoader(object):
    def __init__(self, data, batch_size=50):
        super(CycleLoader, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.iter = self.get_iter()

    def get_iter(self):
        return iter(DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        ))

    def __next__(self):
        next_item = None
        try:
            next_item = next(self.iter)
        except StopIteration:
            self.iter = self.get_iter()
            next_item = next(self.iter)
        finally:
            return next_item
