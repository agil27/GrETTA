import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os

# Define Imagenet Dataset
imagenet_dir = '/n/pfister_lab2/Lab/vcg_natural/imagenet'
imagenetc_dir = '/n/pfister_lab2/Lab/vcg_natural/imagenet-c'
transform_aug = transforms.Compose([
    transforms.RandomAffine(degrees=[-15.0, 15.0], translate=[0.0, 0.2], scale=[0.8, 1.2]),
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trainset = datasets.ImageFolder(
    # os.path.join(imagenet_dir, 'train'),
    imagenet_dir,
    transform=transform_aug
)

train_size = int(0.01 * len(trainset))
trainset, _ = random_split(trainset, [train_size, len(trainset) - train_size])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# clean_valset = datasets.ImageFolder(
#     os.path.join(imagenet_dir, 'val'),
#     transform=transform_aug
# )


def get_train_loader(batch_size):
    return DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )


def get_val_loader(batch_size):
    return DataLoader(
        clean_valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )


# clean_testset = datasets.ImageFolder(
#     os.path.join(imagenet_dir, 'test'),
#     transform_val
# )


# def get_clean_test_loader(batch_size):
#     return DataLoader(
#         clean_testset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=True
#     )


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_corrupt_loader(c, s, batch_size):
    testdir = os.path.join(imagenetc_dir, c, str(s))
    testset = datasets.ImageFolder(testdir, transform_val)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return test_loader
