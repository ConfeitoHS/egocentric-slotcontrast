"""
Discrete VAE (dVAE) from STEVE

Transferred from STEVE repository and adapted for SlotContrast framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    """Gumbel-Softmax sampling"""
    eps = torch.finfo(logits.dtype).tiny
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    """Conv2d with initialization"""
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class Conv2dBlock(nn.Module):
    """Conv2d block with ReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=True, weight_init='kaiming')

    def forward(self, x):
        x = self.m(x)
        return F.relu(x)


class dVAE(nn.Module):
    """
    Discrete VAE from STEVE

    Encodes images into discrete tokens and reconstructs them.
    Used for learning discrete visual representations.
    """

    def __init__(self, vocab_size: int = 4096, img_channels: int = 3, use_checkpoint: bool = False):
        """
        Args:
            vocab_size: Size of discrete vocabulary
            img_channels: Number of image channels
            use_checkpoint: Whether to use gradient checkpointing (saves memory)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.img_channels = img_channels
        self.use_checkpoint = use_checkpoint

        # Encoder: Image -> Discrete tokens
        self.encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 2, 2),  # div by 2
            Conv2dBlock(64, 64, 1, 1), 
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 7, 7),  # div by 7
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1)  # -> vocab_size channels
        )

        # Decoder: Discrete tokens -> Image
        self.decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 7 * 7, 1),
            nn.PixelShuffle(7),  # *7
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),  # *2
            conv2d(64, img_channels, 1),
        )

    def encode(self, images: torch.Tensor, tau: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to discrete tokens

        Args:
            images: Input images (B, C, H, W)
            tau: Temperature for Gumbel-Softmax
            hard: Whether to use hard sampling

        Returns:
            z_logits: Logits for discrete tokens (B, vocab_size, H_enc, W_enc)
            z_soft: Soft discrete tokens (B, vocab_size, H_enc, W_enc)
        """
        z_logits = F.log_softmax(self.encoder(images), dim=1)
        z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)
        return z_logits, z_soft

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens to images

        Args:
            z: Discrete tokens (B, vocab_size, H_enc, W_enc)

        Returns:
            Reconstructed images (B, C, H, W)
        """
        return self.decoder(z)

    def forward(self, images: torch.Tensor, tau: float = 1.0, hard: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full forward pass

        Args:
            images: Input images (B, C, H, W)
            tau: Temperature for Gumbel-Softmax
            hard: Whether to use hard sampling

        Returns:
            Dictionary with:
                - reconstruction: Reconstructed images
                - z_logits: Token logits
                - z_soft: Soft tokens
                - z_hard: Hard tokens
        """
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            z_logits, z_soft = checkpoint(self.encode, images, tau, hard, use_reentrant=False)
        else:
            z_logits, z_soft = self.encode(images, tau, hard)

        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            reconstruction = checkpoint(self.decode, z_soft, use_reentrant=False)
        else:
            reconstruction = self.decode(z_soft)

        return {
            'reconstruction': reconstruction,
            'z_logits': z_logits,
            'z_soft': z_soft,
            'z_hard': z_hard,
        }


class OneHotDictionary(nn.Module):
    """Dictionary for converting one-hot tokens to embeddings"""

    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot tokens (B, N, vocab_size)
        Returns:
            Token embeddings (B, N, emb_size)
        """
        tokens = torch.argmax(x, dim=-1)
        token_embs = self.dictionary(tokens)
        return token_embs


class STEVEDecoder(nn.Module):
    """
    STEVE's Transformer Decoder

    Decodes slots into discrete tokens using transformer.
    Compatible with SlotContrast framework.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 128,
        slot_size: int = 128,
        num_decoder_blocks: int = 4,
        num_decoder_heads: int = 4,
        dropout: float = 0.1,
        image_size: int = 224,
    ):
        """
        Args:
            vocab_size: Size of discrete vocabulary
            d_model: Model dimension
            slot_size: Slot dimension
            num_decoder_blocks: Number of transformer decoder blocks
            num_decoder_heads: Number of attention heads
            dropout: Dropout rate
            image_size: Input image size
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.slot_size = slot_size
        self.image_size = image_size

        # Encoder output size (after 4x downsampling)
        self.enc_size = image_size // 4
        self.seq_len = self.enc_size ** 2

        # Dictionary for token embeddings
        self.dict = OneHotDictionary(vocab_size, d_model)

        # BOS token
        self.bos = nn.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.xavier_uniform_(self.bos)

        # Positional encoding
        self.pos = self._make_positional_encoding(1 + self.seq_len, d_model)

        # Transformer decoder
        from slotcontrast.modules.networks import TransformerDecoder
        self.tf = TransformerDecoder(
            dim=d_model,
            n_blocks=num_decoder_blocks,
            n_heads=num_decoder_heads,
            dropout=dropout,
        )

        # Output head
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Slot projection (if slot_size != d_model)
        if slot_size != d_model:
            self.slot_proj = nn.Linear(slot_size, d_model, bias=False)
        else:
            self.slot_proj = nn.Identity()

    def _make_positional_encoding(self, num_positions: int, d_model: int) -> nn.Module:
        """Create learned positional encoding"""
        class LearnedPE(nn.Module):
            def __init__(self, num_pos, dim):
                super().__init__()
                self.pe = nn.Parameter(torch.zeros(1, num_pos, dim), requires_grad=True)
                nn.init.trunc_normal_(self.pe)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x, offset=0):
                T = x.shape[1]
                return self.dropout(x + self.pe[:, offset:offset + T])

        return LearnedPE(num_positions, d_model)

    def forward(self, slots: torch.Tensor, z_hard: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode slots to discrete tokens

        Args:
            slots: Slot representations (B, num_slots, slot_size)
            z_hard: Hard discrete tokens from dVAE (B, H_enc * W_enc, vocab_size)

        Returns:
            Dictionary with:
                - logits: Predicted token logits (B, H_enc * W_enc, vocab_size)
                - cross_entropy: Cross-entropy loss
        """
        B = slots.shape[0]

        # Project slots
        slots = self.slot_proj(slots)  # (B, num_slots, d_model)

        # Get token embeddings
        z_emb = self.dict(z_hard)  # (B, H_enc * W_enc, d_model)
        # Add BOS token
        z_emb = torch.cat([self.bos.expand(B, -1, -1), z_emb], dim=1)  # (B, 1 + seq_len, d_model)

        # Add positional encoding
        z_emb = self.pos(z_emb)

        # Transformer decode
        print(z_emb.shape, slots.shape)  # Debug print
        pred = self.tf(z_emb[:, :-1], slots)  # (B, seq_len, d_model)
        
        # Predict tokens
        logits = self.head(pred)  # (B, seq_len, vocab_size)

        # Compute cross-entropy loss
        cross_entropy = -(z_hard * torch.log_softmax(logits, dim=-1)).sum() / B

        return {
            'logits': logits,
            'cross_entropy': cross_entropy,
        }


def build_dvae(config) -> dVAE:
    """Build dVAE from config"""
    return dVAE(
        vocab_size=config.get('vocab_size', 4096),
        img_channels=config.get('img_channels', 3),
    )


def build_steve_decoder(config) -> STEVEDecoder:
    """Build STEVE decoder from config"""
    return STEVEDecoder(
        vocab_size=config.get('vocab_size', 4096),
        d_model=config.get('d_model', 128),
        slot_size=config.get('slot_size', 128),
        num_decoder_blocks=config.get('num_decoder_blocks', 4),
        num_decoder_heads=config.get('num_decoder_heads', 4),
        dropout=config.get('dropout', 0.1),
        image_size=config.get('image_size', 224),
    )
