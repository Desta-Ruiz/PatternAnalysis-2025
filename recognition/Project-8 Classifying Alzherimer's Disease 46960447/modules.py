"""
Neural network model definition for Alzheimer's Disease classification.

This module defines the ADNIConvNeXt model, which uses a pretrained ConvNeXt
backbone for transfer learning on brain MRI images.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ADNIConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny based model for AD vs CN binary classification.

    Uses a pretrained ConvNeXt-Tiny backbone (trained on ImageNet) with a custom
    classification head for Alzheimer's Disease detection.

    Architecture:
        - Backbone: ConvNeXt-Tiny (pretrained on ImageNet)
        - Input: 3-channel RGB images of size 224x224
        - Output: 2 logits (one for each class: NC and AD)

    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        freeze_backbone: If True, freeze backbone weights and only train classifier head.
                        This is useful for small datasets to prevent overfitting.
                        Reduces trainable params from ~28M to ~2K.
        dropout: Dropout rate applied before final classifier (default: 0.0)
                Higher values (e.g., 0.3-0.5) help prevent overfitting.

    Example:
        >>> model = ADNIConvNeXt(num_classes=2, freeze_backbone=True, dropout=0.3)
        >>> x = torch.randn(8, 3, 224, 224)  # Batch of 8 images
        >>> logits = model(x)  # Output shape: (8, 2)
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False, dropout: float = 0.0):
        super().__init__()

        # Load pretrained ConvNeXt-Tiny model with ImageNet weights
        # This provides strong feature extraction capabilities from transfer learning
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Optionally freeze backbone parameters for faster training on small datasets
        # When frozen, only the classification head will be trained
        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False  # Prevent gradient updates

        # Replace the final classification layer
        # Original: Linear(768, 1000) for ImageNet's 1000 classes
        # New: Linear(768, 2) for our binary classification (NC vs AD)
        in_dim = self.backbone.classifier[2].in_features  # 768 features

        # Build new classification head
        new_head = []
        if dropout > 0.0:
            # Add dropout for regularization (prevents overfitting)
            new_head.append(nn.Dropout(dropout))
        # Final linear layer maps 768 features to num_classes logits
        new_head.append(nn.Linear(in_dim, num_classes))

        # Replace the classifier
        self.backbone.classifier[2] = nn.Sequential(*new_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            Use softmax/argmax to get class predictions
        """
        return self.backbone(x)
