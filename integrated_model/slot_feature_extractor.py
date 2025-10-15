"""
Slot Feature Extractor

Integrates STEVE (slot generation) with Silicon-Menagerie (pretrained transformers)
to extract object-centric features from egocentric videos.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# Add paths for importing from other repositories
STEVE_PATH = os.path.join(os.path.dirname(__file__), '../../steve')
SILICON_PATH = os.path.join(os.path.dirname(__file__), '../../silicon-menagerie')
sys.path.insert(0, STEVE_PATH)
sys.path.insert(0, SILICON_PATH)

try:
    from steve import STEVE
    from utils import load_model as load_saycam_model
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure STEVE and silicon-menagerie repositories are in the correct locations.")


class SlotFeatureExtractor(nn.Module):
    """
    Integrated model that extracts object-centric features from egocentric videos.

    Architecture:
    1. STEVE encoder: Extracts slots from video frames (object-centric representations)
    2. Pretrained SAYCam transformer: Enriches slot features with learned representations
    3. Feature aggregation: Combines slot and visual features

    Each slot represents either a foreground object or background region in the scene.
    """

    def __init__(
        self,
        steve_checkpoint_path: Optional[str] = None,
        saycam_model_name: str = 'dino_say_vitb14',
        num_slots: int = 7,
        slot_size: int = 128,
        freeze_steve: bool = True,
        freeze_saycam: bool = True,
        feature_fusion: str = 'concat',  # 'concat', 'add', 'attention'
        use_temporal_encoding: bool = True,
    ):
        """
        Args:
            steve_checkpoint_path: Path to pretrained STEVE checkpoint (optional)
            saycam_model_name: Name of pretrained SAYCam model to load
            num_slots: Number of slots for object decomposition
            slot_size: Dimensionality of each slot representation
            freeze_steve: Whether to freeze STEVE encoder weights
            freeze_saycam: Whether to freeze SAYCam transformer weights
            feature_fusion: How to combine slot and visual features
            use_temporal_encoding: Whether to use temporal position encoding
        """
        super().__init__()

        self.num_slots = num_slots
        self.slot_size = slot_size
        self.feature_fusion = feature_fusion
        self.use_temporal_encoding = use_temporal_encoding

        # Initialize STEVE for slot generation
        self.steve_args = self._get_default_steve_args()
        self.steve_args.num_slots = num_slots
        self.steve_args.slot_size = slot_size

        self.steve_model = STEVE(self.steve_args)

        if steve_checkpoint_path and os.path.exists(steve_checkpoint_path):
            self._load_steve_checkpoint(steve_checkpoint_path)

        if freeze_steve:
            for param in self.steve_model.parameters():
                param.requires_grad = False

        # Load pretrained SAYCam transformer
        self.saycam_model = load_saycam_model(saycam_model_name)

        if freeze_saycam:
            for param in self.saycam_model.parameters():
                param.requires_grad = False

        # Get SAYCam feature dimension
        self.saycam_feature_dim = self._get_saycam_feature_dim()

        # Feature fusion layers
        if feature_fusion == 'concat':
            self.fusion_dim = slot_size + self.saycam_feature_dim
            self.fusion_layer = nn.Identity()
        elif feature_fusion == 'add':
            self.fusion_dim = slot_size
            self.projection = nn.Linear(self.saycam_feature_dim, slot_size)
            self.fusion_layer = nn.Identity()
        elif feature_fusion == 'attention':
            self.fusion_dim = slot_size
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=slot_size,
                num_heads=4,
                batch_first=True
            )
            self.projection = nn.Linear(self.saycam_feature_dim, slot_size)
        else:
            raise ValueError(f"Unknown feature_fusion type: {feature_fusion}")

        # Temporal position encoding
        if use_temporal_encoding:
            self.temporal_encoding = nn.Parameter(
                torch.zeros(1, 100, slot_size)  # Max 100 frames
            )
            nn.init.trunc_normal_(self.temporal_encoding, std=0.02)

        # Output projection
        self.output_norm = nn.LayerNorm(self.fusion_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )

    def _get_default_steve_args(self):
        """Get default arguments for STEVE model"""
        class Args:
            def __init__(self):
                self.num_iterations = 3
                self.num_slots = 7
                self.slot_size = 128
                self.mlp_hidden_size = 256
                self.num_predictor_blocks = 2
                self.num_predictor_heads = 4
                self.predictor_dropout = 0.1
                self.img_channels = 3
                self.image_size = 128
                self.cnn_hidden_size = 64
                self.d_model = 128
                self.vocab_size = 4096
                self.num_decoder_blocks = 4
                self.num_decoder_heads = 4
                self.dropout = 0.1

        return Args()

    def _load_steve_checkpoint(self, checkpoint_path: str):
        """Load STEVE checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.steve_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.steve_model.load_state_dict(checkpoint)
        print(f"Loaded STEVE checkpoint from {checkpoint_path}")

    def _get_saycam_feature_dim(self) -> int:
        """Determine the output feature dimension of SAYCam model"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.saycam_model(dummy_input)
        return features.shape[-1]

    def extract_slots(
        self,
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract slots from video using STEVE encoder

        Args:
            video: Input video tensor of shape (B, T, C, H, W)

        Returns:
            slots: Slot representations of shape (B, T, num_slots, slot_size)
            attns_vis: Attention visualizations of shape (B, T, num_slots, C, H, W)
            attns: Raw attention maps of shape (B, T, num_slots, 1, H, W)
        """
        with torch.set_grad_enabled(self.training and not all(
            not p.requires_grad for p in self.steve_model.parameters()
        )):
            slots, attns_vis, attns = self.steve_model.encode(video)

        return slots, attns_vis, attns

    def extract_visual_features(
        self,
        video: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract visual features using pretrained SAYCam transformer

        Args:
            video: Input video tensor of shape (B, T, C, H, W)

        Returns:
            features: Visual features of shape (B, T, feature_dim)
        """
        B, T, C, H, W = video.shape

        # Reshape to process all frames at once
        video_flat = video.view(B * T, C, H, W)

        # Resize to expected input size for transformer (224x224 typically)
        if H != 224 or W != 224:
            video_flat = F.interpolate(
                video_flat, size=(224, 224), mode='bilinear', align_corners=False
            )

        # Extract features
        with torch.set_grad_enabled(self.training and not all(
            not p.requires_grad for p in self.saycam_model.parameters()
        )):
            features = self.saycam_model(video_flat)

        # Reshape back to (B, T, feature_dim)
        features = features.view(B, T, -1)

        return features

    def fuse_features(
        self,
        slots: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse slot representations with visual features

        Args:
            slots: Slot representations of shape (B, T, num_slots, slot_size)
            visual_features: Visual features of shape (B, T, feature_dim)

        Returns:
            fused_features: Combined features of shape (B, T, num_slots, fusion_dim)
        """
        B, T, num_slots, slot_size = slots.shape

        # Expand visual features to match slot dimension
        visual_features_expanded = visual_features.unsqueeze(2).expand(
            B, T, num_slots, -1
        )

        if self.feature_fusion == 'concat':
            # Simple concatenation
            fused = torch.cat([slots, visual_features_expanded], dim=-1)

        elif self.feature_fusion == 'add':
            # Project and add
            visual_projected = self.projection(visual_features_expanded)
            fused = slots + visual_projected

        elif self.feature_fusion == 'attention':
            # Cross-attention between slots and visual features
            visual_projected = self.projection(visual_features_expanded)

            # Reshape for attention
            slots_flat = slots.view(B * T, num_slots, slot_size)
            visual_flat = visual_projected.view(B * T, num_slots, slot_size)

            # Apply cross-attention
            attended, _ = self.cross_attention(
                query=slots_flat,
                key=visual_flat,
                value=visual_flat
            )

            fused = attended.view(B, T, num_slots, slot_size)

        return fused

    def forward(
        self,
        video: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract object-centric features from egocentric video

        Args:
            video: Input video tensor of shape (B, T, C, H, W)
            return_attention: Whether to return attention maps

        Returns:
            Dictionary containing:
                - features: Object-centric features (B, T, num_slots, fusion_dim)
                - slots: Raw slot representations (B, T, num_slots, slot_size)
                - visual_features: SAYCam features (B, T, feature_dim)
                - attention_vis: (optional) Attention visualizations
                - attention_maps: (optional) Raw attention maps
        """
        B, T, C, H, W = video.shape

        # Extract slots from video
        slots, attns_vis, attns = self.extract_slots(video)

        # Extract visual features
        visual_features = self.extract_visual_features(video)

        # Add temporal encoding to slots if enabled
        if self.use_temporal_encoding and T <= self.temporal_encoding.shape[1]:
            temporal_enc = self.temporal_encoding[:, :T, :].unsqueeze(2)
            slots = slots + temporal_enc

        # Fuse features
        fused_features = self.fuse_features(slots, visual_features)

        # Apply output projection
        fused_features = self.output_norm(fused_features)
        fused_features = self.output_projection(fused_features)

        # Prepare output dictionary
        output = {
            'features': fused_features,
            'slots': slots,
            'visual_features': visual_features,
        }

        if return_attention:
            output['attention_vis'] = attns_vis
            output['attention_maps'] = attns

        return output

    def extract_slot_features_only(
        self,
        video: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract only slot features without visual features
        Useful for quick inference

        Args:
            video: Input video tensor of shape (B, T, C, H, W)

        Returns:
            Slot features of shape (B, T, num_slots, slot_size)
        """
        slots, _, _ = self.extract_slots(video)
        return slots

    def get_object_centric_representation(
        self,
        video: torch.Tensor,
        aggregate: str = 'mean'  # 'mean', 'max', 'attention'
    ) -> torch.Tensor:
        """
        Get aggregated object-centric representation for the entire video

        Args:
            video: Input video tensor of shape (B, T, C, H, W)
            aggregate: How to aggregate across time and slots

        Returns:
            Aggregated features of shape (B, fusion_dim)
        """
        output = self.forward(video, return_attention=False)
        features = output['features']  # (B, T, num_slots, fusion_dim)

        if aggregate == 'mean':
            # Average over time and slots
            features = features.mean(dim=[1, 2])
        elif aggregate == 'max':
            # Max pool over time and slots
            features = features.flatten(1, 2).max(dim=1)[0]
        elif aggregate == 'attention':
            # Learnable attention aggregation (would need additional parameters)
            raise NotImplementedError("Attention aggregation not yet implemented")

        return features
