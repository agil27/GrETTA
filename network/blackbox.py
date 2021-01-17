from torchvision.models import resnet50, resnet18
import torch


def resnet50_raw():
    model = resnet50(pretrained=True)
    return model


def resnet18_raw():
    model = resnet18(pretrained=True)
    return model


def resnet50_augmix():
    model = resnet50(pretrained=True)
    model.load_state_dict(torch.load('checkpoints/resnet50_augmix.pth.tar'))
    return model