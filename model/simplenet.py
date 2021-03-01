import torch
import torch.nn as nn
from model.backbone import ResNet18


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv = torch.nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.backbone(x)  # Nx512x7x7
        bb = self.maxpool(x)  # Nx512x3x3
        bb = self.conv(bb)  # Nx4x1x1
        bb = torch.flatten(bb, 1)  # Nx4
        x = self.avgpool(x)  # Nx512x1x1
        x = torch.flatten(x, 1)  # Nx512
        x = self.fc(x)  # Nx1
        x = torch.sigmoid(x)  # Nx1
        y = torch.cat((x, bb), 1)  # Nx5
        return y
