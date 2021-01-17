import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18


def resnet18_policy(num_outputs):
    model = resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_outputs)
    return model


def effnet_b0(num_outputs):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, num_outputs)
    model._swish = nn.Sigmoid()
    return model
