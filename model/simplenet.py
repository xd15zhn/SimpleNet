import torch
import torch.nn as nn
from model.backbone import ResNet18


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.backbone(x)  # Nx512x7x7
        x = self.avgpool(x)  # Nx512x1x1
        x = torch.flatten(x, 1)  # Nx512
        x = self.fc(x)  # Nx5
        return x
