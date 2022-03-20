import torch.nn as nn
from torchvision.models import resnet18, resnet34
from efficientnet_pytorch import EfficientNet
import torch


class Policy(nn.Module):
    def __init__(self, model_name, num_outputs, requires_sigmoid=True):
        super(Policy, self).__init__()
        self.model_name = model_name
        self.requires_sigmoid = requires_sigmoid
        assert model_name in ['resnet18', 'resnet34',
                              'efficientnet-b0', 'efficientnet-b1',
                              'efficientnet-b2', 'efficientnet-b3']

        # define model backbone
        if self.model_name == 'resnet18':
            self.backbone = resnet18(pretrained=False)
        elif self.model_name == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        else:  # efficientnet
            self.backbone = EfficientNet.from_name(model_name)

        # modify output dimensions
        if self.model_name in ['resnet18', 'resnet34']:
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_outputs)
        else:  # efficientnet
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Linear(num_features, num_outputs)

    def forward(self, x):
        outputs = self.backbone(x)
        if self.requires_sigmoid:
            outputs = torch.sigmoid(outputs)
        return outputs

    def identity_init(self):
        if self.model_name in ['resnet18', 'resnet34']:
            self.backbone.fc.weight.data.fill_(0.0)
            if self.requires_sigmoid:
                self.backbone.fc.bias.data.fill_(0.0)
            else:
                self.backbone.fc.bias.data.fill_(0.5)
        else:  # efficientnet
            self.backbone._fc.weight.data.fill_(0.0)
            if self.requires_sigmoid:
                self.backbone._fc.bias.data.fill_(0.0)
            else:
                self.backbone._fc.bias.data.fill_(0.5)