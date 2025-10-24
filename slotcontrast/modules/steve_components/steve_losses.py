"""
STEVE Losses for SlotContrast

Implements STEVE's dVAE reconstruction and cross-entropy losses
compatible with SlotContrast framework.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

from slotcontrast.losses import Loss
from slotcontrast.utils import make_build_fn, read_path


class DVAEReconstructionLoss(Loss):
    """
    dVAE Reconstruction Loss

    MSE loss between original images and dVAE-reconstructed images.
    This is STEVE's image-level reconstruction loss.
    """

    def __init__(
        self,
        pred_key: str = "dvae.reconstruction",
        target_key: str = "video",  # or "image"
        normalize: bool = True,
        reduction: str = "mean",
    ):
        """
        Args:
            pred_key: Key for dVAE reconstruction in outputs
            target_key: Key for target images in inputs
            normalize: Whether to normalize loss by number of elements
            reduction: Reduction mode ('mean', 'sum', or 'none')
        """
        super().__init__(pred_key=pred_key, target_key=target_key)
        self.normalize = normalize
        self.reduction = reduction

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Get dVAE reconstruction from model outputs"""
        return read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        """Get target images from inputs"""
        return read_path(inputs, elements=self.target_path)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss

        Args:
            prediction: Reconstructed images (B, [T], C, H, W)
            target: Original images (B, [T], C, H, W)

        Returns:
            MSE loss scalar
        """
        # Ensure same shape
        assert prediction.shape == target.shape, \
            f"Shape mismatch: pred {prediction.shape} vs target {target.shape}"

        # Compute MSE
        mse = (prediction - target) ** 2

        if self.reduction == 'mean':
            loss = mse.mean()
        elif self.reduction == 'sum':
            loss = mse.sum()
            if self.normalize:
                # Normalize by batch size (and num_frames if video)
                if target.dim() == 5:  # Video
                    loss = loss / (target.shape[0] * target.shape[1])
                else:  # Image
                    loss = loss / target.shape[0]
        else:
            loss = mse

        return loss


class STEVECrossEntropyLoss(Loss):
    """
    STEVE Cross-Entropy Loss

    Cross-entropy loss between predicted discrete tokens and target tokens.
    Used for training the slot-based transformer decoder.
    """

    def __init__(
        self,
        pred_key: str = "steve_decoder.logits",
        target_key: str = "dvae.z_hard",
        reduction: str = "mean",
    ):
        """
        Args:
            pred_key: Key for predicted token logits
            target_key: Key for target discrete tokens (from outputs)
            reduction: Reduction mode
        """
        super().__init__(pred_key=pred_key, target_key=target_key)
        self.reduction = reduction

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Get predicted logits from model outputs"""
        return read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        """Get target tokens from outputs (produced by dVAE)"""
        return read_path(outputs, elements=self.target_path)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss

        Args:
            prediction: Predicted logits (B, [T], seq_len, vocab_size)
            target: Target one-hot tokens (B, [T], seq_len, vocab_size)

        Returns:
            Cross-entropy loss
        """
        # Handle video inputs
        if prediction.dim() == 4:  # (B, T, seq_len, vocab_size)
            B, T = prediction.shape[:2]
            prediction = prediction.reshape(B * T, *prediction.shape[2:])
            target = target.reshape(B * T, *target.shape[2:])

        # Compute cross-entropy
        # target is one-hot, prediction is logits
        log_probs = F.log_softmax(prediction, dim=-1)
        ce_loss = -(target * log_probs).sum(dim=-1)  # (B*T, seq_len)

        if self.reduction == 'mean':
            loss = ce_loss.mean()
        elif self.reduction == 'sum':
            loss = ce_loss.sum() / target.shape[0]
        else:
            loss = ce_loss

        return loss


class STEVEGumbelTemperatureScheduler:
    """
    Scheduler for Gumbel-Softmax temperature

    Anneals temperature from high to low during training.
    Used in STEVE for training the discrete VAE.
    """

    def __init__(
        self,
        start_temp: float = 1.0,
        final_temp: float = 0.1,
        anneal_steps: int = 10000,
        anneal_type: str = 'cosine',  # 'cosine', 'linear', or 'exp'
    ):
        """
        Args:
            start_temp: Initial temperature
            final_temp: Final temperature
            anneal_steps: Number of steps to anneal over
            anneal_type: Type of annealing schedule
        """
        self.start_temp = start_temp
        self.final_temp = final_temp
        self.anneal_steps = anneal_steps
        self.anneal_type = anneal_type

    def get_temperature(self, step: int) -> float:
        """Get temperature for current step"""
        if step >= self.anneal_steps:
            return self.final_temp

        progress = step / self.anneal_steps

        if self.anneal_type == 'cosine':
            import math
            temp = self.final_temp + 0.5 * (self.start_temp - self.final_temp) * \
                   (1 + math.cos(math.pi * progress))
        elif self.anneal_type == 'linear':
            temp = self.start_temp + (self.final_temp - self.start_temp) * progress
        elif self.anneal_type == 'exp':
            import math
            temp = self.start_temp * (self.final_temp / self.start_temp) ** progress
        else:
            raise ValueError(f"Unknown anneal_type: {self.anneal_type}")

        return temp


@make_build_fn(__name__, "loss")
def build_steve_loss(config, name: str):
    """Build STEVE losses from config"""
    if name == "DVAEReconstructionLoss":
        return DVAEReconstructionLoss(
            pred_key=config.get('pred_key', 'dvae.reconstruction'),
            target_key=config.get('target_key', 'video'),
            normalize=config.get('normalize', True),
            reduction=config.get('reduction', 'mean'),
        )
    elif name == "STEVECrossEntropyLoss":
        return STEVECrossEntropyLoss(
            pred_key=config.get('pred_key', 'steve_decoder.logits'),
            target_key=config.get('target_key', 'dvae.z_hard'),
            reduction=config.get('reduction', 'mean'),
        )
    else:
        return None
