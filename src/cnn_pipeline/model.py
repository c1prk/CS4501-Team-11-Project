"""
model.py — DeepfakeCNN

Upgrades over the original:
- Proper MLP classification head (matches the architecture described in
  the presentation: Linear 512->256, ReLU, Dropout, Linear 256->2). The
  original used a single Linear layer which caps expressiveness.
- Supports multiple backbones: resnet18, resnet50, efficientnet_b0.
  ResNet-18 is the default for fair comparison with the original; switch
  to resnet50 for +2-4% at ~2x training cost.
- Exposes freeze/unfreeze/partial-unfreeze helpers so train.py can do
  progressive fine-tuning (freeze -> train head -> unfreeze last block ->
  unfreeze all), which is more stable than unfreezing everything from step 0.
"""
import torch
import torch.nn as nn
from torchvision import models


class DeepfakeCNN(nn.Module):
    def __init__(self, backbone='resnet18', dropout=0.4, freeze_backbone=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = net.fc.in_features  # 512
            net.fc = nn.Identity()
            hidden = 256
        elif backbone == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = net.fc.in_features  # 2048
            net.fc = nn.Identity()
            hidden = 512
        elif backbone == 'efficientnet_b0':
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = net.classifier[1].in_features  # 1280
            net.classifier = nn.Identity()
            hidden = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = net

        # Classification head: matches the architecture described in the slides.
        # Dropout sits between the ReLU and the final Linear to regularize the
        # bottleneck layer, which is the main overfitting site in this setup.
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        if freeze_backbone:
            self.freeze_backbone()

    # --- forward ---
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    # --- freezing controls ---
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_block(self):
        """
        Unfreeze only the last residual block (for ResNets) or last features
        block (EfficientNet). This is the 'progressive unfreezing' step —
        lets the top of the backbone adapt to deepfake artifacts without
        destabilizing the earlier low-level feature extractors.
        """
        if self.backbone_name.startswith('resnet'):
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
        elif self.backbone_name == 'efficientnet_b0':
            # unfreeze the last MBConv block
            last_block = self.backbone.features[-2:]
            for p in last_block.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        """Returns only the params with requires_grad=True — pass to the optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups_for_finetuning(self, head_lr=1e-3, backbone_lr=1e-4):
        """
        Return parameter groups with different LRs — standard trick for
        fine-tuning. The head needs a higher LR (it's randomly initialized)
        while the backbone needs a small LR (already good, just nudge it).
        """
        head_params = list(self.head.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {'params': head_params, 'lr': head_lr},
            {'params': backbone_params, 'lr': backbone_lr},
        ]