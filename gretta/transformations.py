# a set of differentiable image augmentation filters
# powered by opencv kornia
# see https://kornia.readthedocs.io/


import kornia
import torch
import torch.nn.functional as F


# the following transformation function share same paramters
# x: (B, C, H, W) shaped tensor
# v: (B, ) shaped tensor


# geometry
def TranslateX(x, v):
    batch_size = v.size(0)
    translation = torch.zeros((batch_size, 2), device=x.device)
    translation[:, 0] = v
    return kornia.geometry.translate(x, translation)


def TranslateY(x, v):
    batch_size = v.size(0)
    translation = torch.zeros((batch_size, 2), device=x.device)
    translation[:, 1] = v
    return kornia.geometry.translate(x, translation)


def ShearX(x, v):
    batch_size = v.size(0)
    shear = torch.zeros((batch_size, 2), device=x.device)
    shear[:, 0] = v
    return kornia.geometry.shear(x, shear)


def ShearY(x, v):
    batch_size = v.size(0)
    shear = torch.zeros((batch_size, 2), device=x.device)
    shear[:, 1] = v
    return kornia.geometry.shear(x, shear)


def Rotate(x, v):
    return kornia.geometry.rotate(x, v)


def ZoomX(x, v):
    batch_size = v.size(0)
    zoom = torch.ones((batch_size, 2), device=x.device)
    zoom[:, 0] = v
    return kornia.geometry.scale(x, zoom)


def ZoomY(x, v):
    batch_size = v.size(0)
    zoom = torch.ones((batch_size, 2), device=x.device)
    zoom[:, 1] = v
    return kornia.geometry.scale(x, zoom)


# enhance
def Brightness(x, v):
    return kornia.enhance.adjust_brightness(x, v)


def Saturation(x, v):
    return kornia.enhance.adjust_saturation(x, v)


def Hue(x, v):
    return kornia.enhance.adjust_hue(x, v)


def Contrast(x, v):
    return kornia.enhance.adjust_contrast(x, v)


def Gamma(x, v):
    return kornia.enhance.adjust_gamma(x, v)


def sharpen(x, v):
    kernel = torch.tensor([
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    kernel = kernel.to(x.device)
    z = F.conv2d(x, kernel, bias=None, stride=1, groups=x.size(1))
    m = torch.ones_like(z).to(x.device)
    m = F.pad(m, [1, 1, 1, 1])
    z = F.pad(z, [1, 1, 1, 1])
    y = torch.where(m == 1, z, x)
    return torch.stack([x[i] + (y[i] - x[i]) * v[i] for i in range(len(v))])


# filter
def Sharpen(x, v):
    return sharpen(x, torch.abs(v))


# augmentation list
transform_list = [
    (TranslateX, -20, 20),
    (TranslateY, -20, 20),
    (ZoomX, 0.7, 1.3),
    (ZoomY, 0.7, 1.3),
    # (Brightness, -0.3, 0.3),
    (Gamma, 0, 2),
    (Contrast, 0.5, 1.5),
    (Hue, -1, 1)
]

color_transforms = transform_list[4:]
geometry_transforms = transform_list[:4]


def get_tlist(transform):
    if transform == 'color':
        tlist = color_transforms
    elif transform == 'geometry':
        tlist = geometry_transforms
    else:  # full
        tlist = transform_list
    return tlist


def get_num_levels(transform):
    return len(get_tlist(transform))


def apply_transform(img, levels, transform_list=None):
    '''
    :param img: (B, C, H, W) shaped tensor
    :param levels: (B, 4) shaped tensor
    :param filters: image filter subset
    :return: augmented image batches
    '''
    if transform_list is None:
        transform_list = transform_list
    result = img
    for i, (transform, lower, upper) in enumerate(transform_list):
        level = levels[:, i]
        level = torch.clamp(level, 0, 1)
        normalized_level = level * (upper - lower) + lower
        result = transform(result, normalized_level)
    return result
