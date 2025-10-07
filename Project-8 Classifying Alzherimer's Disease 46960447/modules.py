import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ADNIConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny backbone with ImageNet weights and a binary (AD vs CN) classification head.
    Input: 3x224x224
    Output: logits (batch, 2)
    """
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False, dropout: float = 0.0):
        super().__init__()
        # Load pretrained ConvNeXt-Tiny
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Optionally freeze backbone early layers for faster convergence on small datasets
        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False
        # Replace final classifier
        in_dim = self.backbone.classifier[2].in_features
        new_head = []
        if dropout > 0.0:
            new_head.append(nn.Dropout(dropout))
        new_head.append(nn.Linear(in_dim, num_classes))
        self.backbone.classifier[2] = nn.Sequential(*new_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
