# a set of differentiable image augmentation filters
# powered by opencv kornia
# see https://kornia.readthedocs.io/


import kornia
import torch
from gretta.sharpness import sharpness


# the following transformation function share same paramters
# x: (B, C, H, W) shaped tensor
# v: (B, ) shaped tensor


# geometry
def TranslateX(x, v):
    batch_size = v.size(0)
    translation = torch.zeros((batch_size, 2), device=x.device)
    translation[:, 0] = v
    return kornia.translate(x, translation)


def TranslateY(x, v):
    batch_size = v.size(0)
    translation = torch.zeros((batch_size, 2), device=x.device)
    translation[:, 1] = v
    return kornia.translate(x, translation)


def ShearX(x, v):
    batch_size = v.size(0)
    shear = torch.zeros((batch_size, 2), device=x.device)
    shear[:, 0] = v
    return kornia.shear(x, shear)


def ShearY(x, v):
    batch_size = v.size(0)
    shear = torch.zeros((batch_size, 2), device=x.device)
    shear[:, 1] = v
    return kornia.shear(x, shear)


def Rotate(x, v):
    return kornia.rotate(x, v)


def ZoomX(x, v):
    batch_size = v.size(0)
    zoom = torch.ones((batch_size, 2), device=x.device)
    zoom[:, 0] = v
    return kornia.scale(x, zoom)


def ZoomY(x, v):
    batch_size = v.size(0)
    zoom = torch.ones((batch_size, 2), device=x.device)
    zoom[:, 1] = v
    return kornia.scale(x, zoom)


# enhance
def Brightness(x, v):
    return kornia.adjust_brightness(x, v)


def Saturation(x, v):
    return kornia.adjust_saturation(x, v)


def Hue(x, v):
    return kornia.adjust_hue(x, v)


def Contrast(x, v):
    return kornia.adjust_contrast(x, v)


def Gamma(x, v):
    return kornia.adjust_gamma(x, v)


# filter
def Sharpness(x, v):
    return sharpness(x, torch.abs(v))


# augmentation list
transform_list = [
    (TranslateX, -20, 20),
    (TranslateY, -20, 20),
    (ZoomX, 0.7, 1.3),
    (ZoomY, 0.7, 1.3),
    (Brightness, -0.3, 0.3),
    (Contrast, 0.8, 1.2),
    (Sharpness, -0.3, 0.3),
]

color_transforms = transform_list[4:]
geometry_transforms = transform_list[:4]


def apply_transform(img, levels):
    '''
    :param img: (B, C, H, W) shaped tensor
    :param levels: (B, 4) shaped tensor
    :param filters: image filter subset
    :return: augmented image batches
    '''
    result = img
    for i, (transform, lower, upper) in enumerate(transform_list):
        level = levels[:, i]
        normalized_level = level * (upper - lower) + lower
        result = transform(result, normalized_level)
    return result
