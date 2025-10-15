"""
Custom Encoder Module for Training

Integrates silicon-menagerie pretrained ViT as backbone encoder
for the SlotContrast training framework.
"""

import sys
import os
from typing import Dict, Optional
import torch
from torch import nn

# Add silicon-menagerie to path
SILICON_PATH = os.path.join(os.path.dirname(__file__), '../../../silicon-menagerie')
sys.path.insert(0, SILICON_PATH)

try:
    from utils import load_model as load_saycam_model
except ImportError:
    print("Warning: Could not import silicon-menagerie utils")


class SiliconMenagerieEncoder(nn.Module):
    """
    Encoder using pretrained ViT from silicon-menagerie

    This wraps the pretrained model and extracts features in a format
    compatible with SlotContrast framework.
    """

    def __init__(
        self,
        model_name: str = 'dino_say_vitb14',
        frozen: bool = True,
        output_layer: str = 'last',  # 'last', 'intermediate', 'all'
        extract_cls_token: bool = False,
    ):
        """
        Args:
            model_name: Name of pretrained model from silicon-menagerie
            frozen: Whether to freeze the pretrained weights
            output_layer: Which layer features to extract
            extract_cls_token: Whether to include CLS token in output
        """
        super().__init__()

        self.model_name = model_name
        self.frozen = frozen
        self.output_layer = output_layer
        self.extract_cls_token = extract_cls_token

        # Load pretrained model
        self.backbone = load_saycam_model(model_name)

        # Freeze if requested
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Determine feature dimension
        self._determine_feature_dim()

    def _determine_feature_dim(self):
        """Determine output feature dimension"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(dummy_input)
            else:
                features = self.backbone(dummy_input)

        if features.dim() == 3:
            self.feature_dim = features.shape[-1]
            self.is_patch_features = True
        else:
            self.feature_dim = features.shape[-1]
            self.is_patch_features = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            images: Input images of shape (B, C, H, W)

        Returns:
            Dictionary with keys:
                - features: Main features (B, N, D) where N is number of patches
                - backbone_features: Same as features (for compatibility)
        """
        B, C, H, W = images.shape

        # Extract features
        if self.frozen:
            with torch.no_grad():
                if hasattr(self.backbone, 'forward_features'):
                    features = self.backbone.forward_features(images)
                else:
                    # For models that don't have forward_features
                    features = self.backbone(images)
        else:
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(images)
            else:
                features = self.backbone(images)

        # Handle different output formats
        if features.dim() == 2:
            # Global pooled features (B, D) -> (B, 1, D)
            features = features.unsqueeze(1)
        elif features.dim() == 3:
            # Patch features (B, N+1, D) with CLS token
            if not self.extract_cls_token:
                # Remove CLS token (first token)
                features = features[:, 1:, :]
        elif features.dim() == 4:
            # Spatial features (B, D, H, W) -> (B, H*W, D)
            B, D, H_feat, W_feat = features.shape
            features = features.flatten(2).transpose(1, 2)

        return {
            'features': features,
            'backbone_features': features.clone(),
        }

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval if frozen"""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self


def build_silicon_encoder(config) -> SiliconMenagerieEncoder:
    """
    Build encoder from config dict

    Config format:
        model_name: str (e.g., 'dino_say_vitb14')
        frozen: bool (default: True)
        output_layer: str (default: 'last')
        extract_cls_token: bool (default: False)
    """
    return SiliconMenagerieEncoder(
        model_name=config.get('model_name', 'dino_say_vitb14'),
        frozen=config.get('frozen', True),
        output_layer=config.get('output_layer', 'last'),
        extract_cls_token=config.get('extract_cls_token', False),
    )
