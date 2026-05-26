# backend/model.py
"""
PyTorch model for blood group detection from fingerprints.

Mirrors the prior Keras design:
  - EfficientNet-B0 backbone (ImageNet weights) via torchvision.
  - 1x1 Conv grayscale → 3 channels + BatchNorm before backbone.
  - Phase 1: backbone frozen; phase 2: last two feature blocks unfrozen.
  - Head: global pool → BN → Linear(256) → Dropout(0.5) → Linear(128) → Dropout(0.4) → logits (4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import INPUT_SIZE, NUM_CLASSES


class HemaTraceNet(nn.Module):
    """Single module for both training phases; use set_phase() to freeze/unfreeze."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        H, W = INPUT_SIZE
        self._input_hw = (H, W)

        self.gray = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
        )

        body = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = body.features
        feat_dim = 1280  # EfficientNet-B0 last feature map channels

        self.head_bn = nn.BatchNorm1d(feat_dim)
        self.fc1 = nn.Linear(feat_dim, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (N, 1, H, W)
        x = self.gray(x)
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head_bn(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

    def set_phase(self, phase: int):
        """phase 1: freeze backbone. phase 2: unfreeze last two backbone blocks + adapter + head."""
        if phase == 1:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.gray.parameters():
                p.requires_grad = True
            for p in self.head_bn.parameters():
                p.requires_grad = True
            for p in self.fc1.parameters():
                p.requires_grad = True
            for p in self.fc2.parameters():
                p.requires_grad = True
            for p in self.fc3.parameters():
                p.requires_grad = True
        elif phase == 2:
            n_blocks = len(self.backbone)
            for i, block in enumerate(self.backbone):
                train = i >= max(0, n_blocks - 2)
                for p in block.parameters():
                    p.requires_grad = train
            for p in self.gray.parameters():
                p.requires_grad = True
            for p in self.head_bn.parameters():
                p.requires_grad = True
            for p in self.fc1.parameters():
                p.requires_grad = True
            for p in self.fc2.parameters():
                p.requires_grad = True
            for p in self.fc3.parameters():
                p.requires_grad = True
        else:
            raise ValueError("phase must be 1 or 2")


def build_model(phase: int = 1):
    """
    Build model for a training phase.
    Returns: (model, backbone_module) — backbone is model.backbone for symmetry with old code.
    """
    model = HemaTraceNet()
    model.set_phase(phase)
    return model, model.backbone


def print_model_summary(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    H, W = INPUT_SIZE
    print(f"\n📐 Model summary (PyTorch):")
    print(f"   Total params     : {total:,}")
    print(f"   Trainable params : {trainable:,}")
    print(f"   Frozen params    : {frozen:,}")
    print(f"   Input shape      : (1, {H}, {W})")
    print(f"   Output classes   : {NUM_CLASSES}")


if __name__ == "__main__":
    print("=== Phase 1 (frozen base) ===")
    m1, _ = build_model(phase=1)
    print_model_summary(m1)

    print("\n=== Phase 2 (last blocks unfrozen) ===")
    m2, _ = build_model(phase=2)
    print_model_summary(m2)
