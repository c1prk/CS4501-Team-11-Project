import torch
import torch.nn as nn
from torchvision import models

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        # load ResNet18 pretrained on ImageNet
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        # We replace it with one that outputs 2 classes — real or fake
        in_features = self.backbone.fc.in_features 
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)