"""
Silicon-Menagerie ViT Backbone

Integrates pretrained models from silicon-menagerie as encoder backbones.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional
import torch
from torch import nn

# Add silicon-menagerie to path
SILICON_PATH = Path(__file__).parent.parent.parent.parent / 'silicon-menagerie'
if SILICON_PATH.exists():
    sys.path.insert(0, str(SILICON_PATH))

try:
    from utils import load_model as load_saycam_model
    SILICON_AVAILABLE = True
except ImportError:
    SILICON_AVAILABLE = False
    print("Warning: silicon-menagerie not available. Install or check path.")


class SiliconViTBackbone(nn.Module):
    """
    ViT backbone from silicon-menagerie pretrained models

    Compatible with SlotContrast's encoder framework.
    Extracts patch features from pretrained ViT models trained on SAYCam.
    """

    def __init__(
        self,
        model_name: str = 'dino_say_vitb14',
        frozen: bool = True,
        extract_cls_token: bool = False,
        layer_index: Optional[int] = None,
    ):
        """
        Args:
            model_name: Name of pretrained model (e.g., 'dino_say_vitb14')
            frozen: Whether to freeze the backbone
            extract_cls_token: Whether to include CLS token in output
            layer_index: Which layer to extract features from (None = last layer)
        """
        super().__init__()

        if not SILICON_AVAILABLE:
            raise ImportError(
                "silicon-menagerie not available. "
                "Ensure repository exists at: " + str(SILICON_PATH)
            )

        self.model_name = model_name
        self.frozen = frozen
        self.extract_cls_token = extract_cls_token
        self.layer_index = layer_index

        # Load pretrained model
        print(f"Loading silicon-menagerie model: {model_name}")
        self.backbone = load_saycam_model(model_name)

        # Freeze if requested
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Determine output dimension
        self._determine_output_dim()

    def _determine_output_dim(self):
        """Determine the output feature dimension"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self._extract_features(dummy_input)
        self.feature_dim = features.shape[-1]
        self.num_patches = features.shape[1]

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features from ViT"""
        if hasattr(self.backbone, 'forward_features'):
            # For ViT models with forward_features method
            features = self.backbone.forward_features(images)
        else:
            # Fallback for models without forward_features
            features = self.backbone(images)

        # Handle different output formats
        if features.dim() == 2:
            # Global features (B, D) -> expand to (B, 1, D)
            features = features.unsqueeze(1)
        elif features.dim() == 3:
            # Patch features (B, N+1, D) with CLS token
            if not self.extract_cls_token and features.shape[1] > 1:
                # Remove CLS token (first token)
                features = features[:, 1:, :]
        elif features.dim() == 4:
            # Spatial features (B, D, H, W) -> (B, H*W, D)
            B, D, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Patch features (B, num_patches, feature_dim)
        """
        if self.frozen:
            with torch.no_grad():
                features = self._extract_features(images)
        else:
            features = self._extract_features(images)

        return features

    def train(self, mode: bool = True):
        """Override train to keep backbone in eval if frozen"""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self


class SiliconViTExtractor(nn.Module):
    """
    Feature extractor using silicon-menagerie ViT

    Returns features in the format expected by SlotContrast framework.
    Compatible with TimmExtractor interface.
    """

    def __init__(
        self,
        model: str = 'dino_say_vitb14',
        frozen: bool = True,
        extract_cls_token: bool = False,
        features: Optional[list] = None,  # For compatibility with TimmExtractor
    ):
        """
        Args:
            model: Model name from silicon-menagerie
            frozen: Whether to freeze weights
            extract_cls_token: Whether to include CLS token
            features: Feature names to extract (for compatibility, not used)
        """
        super().__init__()

        self.backbone = SiliconViTBackbone(
            model_name=model,
            frozen=frozen,
            extract_cls_token=extract_cls_token,
        )

        self.feature_dim = self.backbone.feature_dim
        self.frozen = frozen

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with SlotContrast's encoder framework

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Dictionary with feature keys compatible with SlotContrast
        """
        features = self.backbone(images)

        # Return in format expected by FrameEncoder
        # The features should be (B, num_tokens, feature_dim)
        return {
            'vit_block12': features,  # Main features
            'vit_block_keys12': features.clone(),  # Copy for compatibility
        }


def build_silicon_vit(config) -> SiliconViTExtractor:
    """
    Build silicon-menagerie ViT from config

    Config format compatible with SlotContrast:
        model: str (e.g., 'dino_say_vitb14')
        frozen: bool (default: True)
        extract_cls_token: bool (default: False)
        features: list (optional, for compatibility)
    """
    return SiliconViTExtractor(
        model=config.get('model', 'dino_say_vitb14'),
        frozen=config.get('frozen', True),
        extract_cls_token=config.get('extract_cls_token', False),
        features=config.get('features', None),
    )


# Register with SlotContrast's build system
def register_silicon_vit():
    """Register SiliconViTExtractor with SlotContrast encoders"""
    import slotcontrast.modules.encoders as encoders_module

    # Add to encoders module
    if not hasattr(encoders_module, 'SiliconViTExtractor'):
        encoders_module.SiliconViTExtractor = SiliconViTExtractor
        print("Registered SiliconViTExtractor with SlotContrast encoders")


# Auto-register when module is imported
if SILICON_AVAILABLE:
    try:
        register_silicon_vit()
    except Exception as e:
        print(f"Warning: Could not register SiliconViTExtractor: {e}")
