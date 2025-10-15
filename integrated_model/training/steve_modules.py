"""
STEVE-Compatible Modules

Provides STEVE-style slot attention that's compatible with
SlotContrast training framework.
"""

import sys
import os
from pathlib import Path
import torch
from torch import nn
from typing import Dict, Optional, Tuple

# Add STEVE to path
STEVE_PATH = Path(__file__).parent.parent.parent / 'steve'
sys.path.insert(0, str(STEVE_PATH))

try:
    from steve import SlotAttentionVideo
    from utils import gru_cell, linear
except ImportError:
    print("Warning: Could not import STEVE modules")


class STEVESlotAttentionAdapter(nn.Module):
    """
    Adapter that wraps STEVE's SlotAttentionVideo to work with SlotContrast framework

    This allows us to use STEVE's slot attention mechanism within
    the SlotContrast training pipeline.
    """

    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        num_iterations: int = 3,
        num_slots: int = 7,
        mlp_hidden_size: Optional[int] = None,
        num_predictor_blocks: int = 1,
        num_predictor_heads: int = 4,
        predictor_dropout: float = 0.1,
        epsilon: float = 1e-8,
        frozen: bool = False,
    ):
        """
        Args:
            inp_dim: Input feature dimension
            slot_dim: Slot representation dimension
            num_iterations: Number of slot attention iterations
            num_slots: Number of slots
            mlp_hidden_size: Hidden size for MLP (default: 4 * slot_dim)
            num_predictor_blocks: Number of transformer blocks for predictor
            num_predictor_heads: Number of attention heads in predictor
            predictor_dropout: Dropout for predictor
            epsilon: Small constant for numerical stability
            frozen: Whether to freeze parameters
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.inp_dim = inp_dim
        self.slot_dim = slot_dim
        self.epsilon = epsilon

        if mlp_hidden_size is None:
            mlp_hidden_size = 4 * slot_dim

        # Slot initialization parameters (shared across all slots)
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # Normalization layers
        self.norm_inputs = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Attention projections
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(inp_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(inp_dim, slot_dim, bias=False)

        # Slot update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_dim)
        )

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def initialize_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize slots with learned Gaussian parameters"""
        slots = torch.randn(batch_size, self.num_slots, self.slot_dim, device=device)
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots
        return slots

    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
        n_iters: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply slot attention

        Args:
            slots: Initial slot representations (B, num_slots, slot_dim)
            features: Input features (B, num_features, inp_dim)
            n_iters: Number of iterations (default: self.num_iterations)

        Returns:
            Dictionary with:
                - slots: Updated slot representations (B, num_slots, slot_dim)
                - masks: Attention masks (B, num_slots, num_features)
        """
        if n_iters is None:
            n_iters = self.num_iterations

        batch_size = features.shape[0]

        # Normalize inputs
        features = self.norm_inputs(features)

        # Project to keys and values
        k = self.project_k(features)  # (B, num_features, slot_dim)
        v = self.project_v(features)  # (B, num_features, slot_dim)
        k = k * (self.slot_dim ** -0.5)

        # Iterative slot attention
        for _ in range(n_iters):
            slots_prev = slots

            # Normalize slots
            slots_norm = self.norm_slots(slots)

            # Compute attention
            q = self.project_q(slots_norm)  # (B, num_slots, slot_dim)
            attn_logits = torch.bmm(k, q.transpose(-1, -2))  # (B, num_features, num_slots)
            attn_vis = torch.softmax(attn_logits, dim=-1)

            # Weighted mean with normalization
            attn = attn_vis + self.epsilon
            attn = attn / attn.sum(dim=1, keepdim=True)

            # Aggregate
            updates = torch.bmm(attn.transpose(-1, -2), v)  # (B, num_slots, slot_dim)

            # GRU update
            slots = self.gru(
                updates.flatten(0, 1),
                slots_prev.flatten(0, 1)
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_dim)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        return {
            'slots': slots,
            'masks': attn_vis.transpose(-1, -2),  # (B, num_slots, num_features)
        }


class TemporalSlotAttention(nn.Module):
    """
    Temporal slot attention that processes video frames sequentially

    Compatible with SlotContrast's video processing pipeline
    """

    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        num_iterations: int = 3,
        num_slots: int = 7,
        mlp_hidden_size: Optional[int] = None,
        frozen: bool = False,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Slot attention module
        self.slot_attention = STEVESlotAttentionAdapter(
            inp_dim=inp_dim,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            num_slots=num_slots,
            mlp_hidden_size=mlp_hidden_size,
            frozen=frozen,
        )

    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
        n_iters: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process features with slot attention

        For compatibility with SlotContrast, this has the same interface
        as SlotContrast's SlotAttention module.

        Args:
            slots: Initial slots (B, num_slots, slot_dim)
            features: Input features (B, num_features, inp_dim)
            n_iters: Number of iterations

        Returns:
            Dictionary with slots and masks
        """
        return self.slot_attention(slots, features, n_iters)


def build_steve_slot_attention(config) -> TemporalSlotAttention:
    """
    Build STEVE-style slot attention from config

    Config format:
        inp_dim: int
        slot_dim: int
        n_iters: int (default: 3)
        num_slots: int (default: 7)
        mlp_hidden_size: int (optional)
        frozen: bool (default: False)
    """
    return TemporalSlotAttention(
        inp_dim=config.get('inp_dim'),
        slot_dim=config.get('slot_dim'),
        num_iterations=config.get('n_iters', 3),
        num_slots=config.get('num_slots', 7),
        mlp_hidden_size=config.get('mlp_hidden_size'),
        frozen=config.get('frozen', False),
    )
