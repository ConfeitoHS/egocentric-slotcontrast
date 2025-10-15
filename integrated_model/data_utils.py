"""
Data Loading Utilities

Utilities for loading and preprocessing egocentric videos for the integrated model
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from pathlib import Path


class EgocentricVideoDataset(Dataset):
    """
    Dataset for loading egocentric videos

    Supports various video formats and preprocessing options
    """

    def __init__(
        self,
        video_paths: Union[List[str], str],
        labels: Optional[Union[List[int], np.ndarray]] = None,
        num_frames: int = 8,
        frame_size: Tuple[int, int] = (128, 128),
        frame_stride: int = 1,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        temporal_sampling: str = 'uniform',  # 'uniform', 'random', 'consecutive'
    ):
        """
        Args:
            video_paths: List of paths to video files or directory containing videos
            labels: Optional labels for each video
            num_frames: Number of frames to sample from each video
            frame_size: Target size for each frame (H, W)
            frame_stride: Stride for temporal sampling
            transform: Optional transform to apply to each frame
            normalize: Whether to normalize frames to [0, 1]
            temporal_sampling: How to sample frames from video
        """
        super().__init__()

        # Handle directory input
        if isinstance(video_paths, str):
            if os.path.isdir(video_paths):
                video_paths = self._find_videos_in_directory(video_paths)
            else:
                video_paths = [video_paths]

        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.transform = transform
        self.normalize = normalize
        self.temporal_sampling = temporal_sampling

        # Check if labels match videos
        if labels is not None:
            assert len(labels) == len(video_paths), \
                f"Number of labels ({len(labels)}) must match number of videos ({len(video_paths)})"

    def _find_videos_in_directory(self, directory: str) -> List[str]:
        """Find all video files in directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_paths = []

        for ext in video_extensions:
            video_paths.extend(Path(directory).rglob(f'*{ext}'))

        return [str(p) for p in sorted(video_paths)]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess video

        Returns:
            Dictionary containing:
                - video: Tensor of shape (T, C, H, W)
                - label: Label (if available)
                - path: Path to video file
        """
        video_path = self.video_paths[idx]

        # Load video
        video = self._load_video(video_path)

        # Apply transform if provided
        if self.transform is not None:
            video = self.transform(video)

        output = {
            'video': video,
            'path': video_path,
        }

        if self.labels is not None:
            output['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return output

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load video from file

        Returns:
            video: Tensor of shape (T, C, H, W)
        """
        try:
            # Try using torchvision.io first
            import torchvision.io as io
            video, audio, info = io.read_video(video_path, pts_unit='sec')

            # video shape: (T, H, W, C)
            video = video.permute(0, 3, 1, 2)  # -> (T, C, H, W)
            video = video.float()

            if self.normalize:
                video = video / 255.0

        except ImportError:
            # Fallback to opencv
            import cv2

            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise ValueError(f"Could not load video: {video_path}")

            video = np.stack(frames, axis=0)
            video = torch.from_numpy(video).float()
            video = video.permute(0, 3, 1, 2)  # -> (T, C, H, W)

            if self.normalize:
                video = video / 255.0

        # Sample frames
        video = self._sample_frames(video)

        # Resize frames
        video = self._resize_frames(video)

        return video

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        """
        Sample frames from video

        Args:
            video: Input video of shape (T, C, H, W)

        Returns:
            Sampled video of shape (num_frames, C, H, W)
        """
        T, C, H, W = video.shape

        if T < self.num_frames:
            # Repeat frames if video is too short
            repeat_factor = (self.num_frames + T - 1) // T
            video = video.repeat(repeat_factor, 1, 1, 1)[:self.num_frames]
            return video

        if self.temporal_sampling == 'uniform':
            # Uniformly sample frames
            indices = torch.linspace(0, T - 1, self.num_frames).long()

        elif self.temporal_sampling == 'random':
            # Randomly sample frames
            indices = torch.randperm(T)[:self.num_frames].sort()[0]

        elif self.temporal_sampling == 'consecutive':
            # Sample consecutive frames
            max_start = max(0, T - self.num_frames * self.frame_stride)
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            indices = torch.arange(
                start_idx,
                start_idx + self.num_frames * self.frame_stride,
                self.frame_stride
            )[:self.num_frames]

        else:
            raise ValueError(f"Unknown temporal sampling: {self.temporal_sampling}")

        return video[indices]

    def _resize_frames(self, video: torch.Tensor) -> torch.Tensor:
        """
        Resize video frames

        Args:
            video: Input video of shape (T, C, H, W)

        Returns:
            Resized video of shape (T, C, target_H, target_W)
        """
        import torch.nn.functional as F

        T, C, H, W = video.shape
        target_H, target_W = self.frame_size

        if H != target_H or W != target_W:
            video = F.interpolate(
                video,
                size=self.frame_size,
                mode='bilinear',
                align_corners=False
            )

        return video


def load_egocentric_video(
    video_path: str,
    num_frames: int = 8,
    frame_size: Tuple[int, int] = (128, 128),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convenience function to load a single egocentric video

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        frame_size: Target size for each frame (H, W)
        normalize: Whether to normalize to [0, 1]

    Returns:
        video: Tensor of shape (1, T, C, H, W)
    """
    dataset = EgocentricVideoDataset(
        video_paths=[video_path],
        num_frames=num_frames,
        frame_size=frame_size,
        normalize=normalize,
    )

    video_dict = dataset[0]
    video = video_dict['video'].unsqueeze(0)  # Add batch dimension

    return video


class VideoTransform:
    """
    Composable video transformations
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            video = transform(video)
        return video


class RandomHorizontalFlip:
    """Randomly flip video horizontally"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return video.flip(-1)  # Flip width dimension
        return video


class RandomCrop:
    """Randomly crop video frames"""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video.shape
        target_H, target_W = self.size

        if H < target_H or W < target_W:
            # Pad if necessary
            pad_H = max(0, target_H - H)
            pad_W = max(0, target_W - W)
            video = torch.nn.functional.pad(
                video, (0, pad_W, 0, pad_H), mode='constant', value=0
            )
            H, W = video.shape[-2:]

        # Random crop
        top = torch.randint(0, H - target_H + 1, (1,)).item()
        left = torch.randint(0, W - target_W + 1, (1,)).item()

        return video[:, :, top:top + target_H, left:left + target_W]


class CenterCrop:
    """Center crop video frames"""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video.shape
        target_H, target_W = self.size

        if H < target_H or W < target_W:
            raise ValueError(f"Video size ({H}, {W}) is smaller than target size {self.size}")

        top = (H - target_H) // 2
        left = (W - target_W) // 2

        return video[:, :, top:top + target_H, left:left + target_W]


class ColorJitter:
    """Apply color jittering to video frames"""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # Apply same jitter to all frames
        brightness_factor = 1.0 + torch.empty(1).uniform_(
            -self.brightness, self.brightness
        ).item()
        contrast_factor = 1.0 + torch.empty(1).uniform_(
            -self.contrast, self.contrast
        ).item()

        # Brightness
        video = video * brightness_factor

        # Contrast
        mean = video.mean(dim=[1, 2, 3], keepdim=True)
        video = (video - mean) * contrast_factor + mean

        # Clamp to [0, 1] if normalized
        video = torch.clamp(video, 0, 1)

        return video


class Normalize:
    """Normalize video with mean and std"""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return (video - self.mean) / self.std


def create_dataloader(
    video_paths: Union[List[str], str],
    labels: Optional[Union[List[int], np.ndarray]] = None,
    batch_size: int = 8,
    num_frames: int = 8,
    frame_size: Tuple[int, int] = (128, 128),
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for egocentric videos

    Args:
        video_paths: List of video paths or directory
        labels: Optional labels
        batch_size: Batch size
        num_frames: Number of frames per video
        frame_size: Target frame size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        transform: Optional transform
        **kwargs: Additional arguments for Dataset

    Returns:
        DataLoader instance
    """
    dataset = EgocentricVideoDataset(
        video_paths=video_paths,
        labels=labels,
        num_frames=num_frames,
        frame_size=frame_size,
        transform=transform,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
