"""
Visualization Utilities

Tools for visualizing slots, attention maps, and features
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def visualize_slots(
    video: torch.Tensor,
    slots: torch.Tensor,
    attention_maps: torch.Tensor,
    save_path: str = 'slots_visualization.png',
    max_frames: int = 8,
    max_slots: int = 7,
):
    """
    Visualize slots and their attention maps

    Args:
        video: Video tensor of shape (B, T, C, H, W)
        slots: Slot representations of shape (B, T, num_slots, slot_size)
        attention_maps: Attention maps of shape (B, T, num_slots, 1, H, W)
        save_path: Path to save visualization
        max_frames: Maximum number of frames to visualize
        max_slots: Maximum number of slots to visualize
    """
    B, T, C, H, W = video.shape
    num_slots = min(slots.shape[2], max_slots)
    num_frames = min(T, max_frames)

    # Select first video in batch
    video = video[0, :num_frames].cpu().numpy()
    attention_maps = attention_maps[0, :num_frames, :num_slots, 0].cpu().numpy()

    # Denormalize video if needed
    if video.max() <= 1.0:
        video = video
    else:
        video = video / 255.0

    # Create figure
    fig, axes = plt.subplots(
        num_slots + 1, num_frames,
        figsize=(num_frames * 2, (num_slots + 1) * 2)
    )

    if num_frames == 1:
        axes = axes.reshape(-1, 1)

    # Plot original frames
    for t in range(num_frames):
        frame = video[t].transpose(1, 2, 0)  # CHW -> HWC
        axes[0, t].imshow(frame)
        axes[0, t].set_title(f'Frame {t}')
        axes[0, t].axis('off')

    # Plot attention maps for each slot
    for s in range(num_slots):
        for t in range(num_frames):
            # Get attention map
            attn = attention_maps[t, s]

            # Overlay attention on frame
            frame = video[t].transpose(1, 2, 0)
            axes[s + 1, t].imshow(frame)
            axes[s + 1, t].imshow(attn, alpha=0.6, cmap='jet')

            if t == 0:
                axes[s + 1, t].set_ylabel(f'Slot {s}', fontsize=12)
            axes[s + 1, t].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {save_path}")


def visualize_attention_evolution(
    attention_maps: torch.Tensor,
    save_path: str = 'attention_evolution.png',
    slot_idx: int = 0,
):
    """
    Visualize how attention for a single slot evolves over time

    Args:
        attention_maps: Attention maps of shape (B, T, num_slots, 1, H, W)
        save_path: Path to save visualization
        slot_idx: Which slot to visualize
    """
    # Select first video and target slot
    attn = attention_maps[0, :, slot_idx, 0].cpu().numpy()  # (T, H, W)
    T, H, W = attn.shape

    # Create figure
    fig, axes = plt.subplots(1, T, figsize=(T * 2, 2))
    if T == 1:
        axes = [axes]

    for t in range(T):
        axes[t].imshow(attn[t], cmap='jet')
        axes[t].set_title(f't={t}')
        axes[t].axis('off')

    plt.suptitle(f'Attention Evolution for Slot {slot_idx}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved attention evolution to {save_path}")


def visualize_slot_features(
    slots: torch.Tensor,
    save_path: str = 'slot_features.png',
    method: str = 'pca',
):
    """
    Visualize slot features using dimensionality reduction

    Args:
        slots: Slot representations of shape (B, T, num_slots, slot_size)
        save_path: Path to save visualization
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    B, T, num_slots, slot_size = slots.shape

    # Flatten to (B*T*num_slots, slot_size)
    slots_flat = slots.reshape(-1, slot_size).cpu().numpy()

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        slots_2d = reducer.fit_transform(slots_flat)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        slots_2d = reducer.fit_transform(slots_flat)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reshape back
    slots_2d = slots_2d.reshape(B, T, num_slots, 2)

    # Plot first video
    plt.figure(figsize=(10, 8))

    for s in range(num_slots):
        slot_trajectory = slots_2d[0, :, s]  # (T, 2)
        plt.plot(
            slot_trajectory[:, 0],
            slot_trajectory[:, 1],
            'o-',
            label=f'Slot {s}',
            alpha=0.7
        )

        # Mark start and end
        plt.scatter(
            slot_trajectory[0, 0],
            slot_trajectory[0, 1],
            c='green',
            s=100,
            marker='*'
        )
        plt.scatter(
            slot_trajectory[-1, 0],
            slot_trajectory[-1, 1],
            c='red',
            s=100,
            marker='X'
        )

    plt.legend()
    plt.title(f'Slot Features Trajectory ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved feature visualization to {save_path}")


def visualize_feature_similarity(
    features: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    save_path: str = 'feature_similarity.png',
):
    """
    Visualize similarity matrix of features

    Args:
        features: Features of shape (N, feature_dim)
        labels: Optional labels of shape (N,)
        save_path: Path to save visualization
    """
    # Compute similarity matrix
    features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
    similarity = torch.mm(features_norm, features_norm.t()).cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(similarity, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Add labels if provided
    if labels is not None:
        labels_np = labels.cpu().numpy()
        unique_labels = np.unique(labels_np)

        # Draw lines to separate classes
        for i in range(1, len(labels_np)):
            if labels_np[i] != labels_np[i - 1]:
                ax.axhline(i - 0.5, color='red', linestyle='--', linewidth=1)
                ax.axvline(i - 0.5, color='red', linestyle='--', linewidth=1)

    ax.set_title('Feature Similarity Matrix')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved similarity matrix to {save_path}")


def create_video_from_frames(
    frames: np.ndarray,
    save_path: str = 'output_video.mp4',
    fps: int = 10,
):
    """
    Create video from frames

    Args:
        frames: Frames of shape (T, H, W, C) in range [0, 1]
        save_path: Path to save video
        fps: Frames per second
    """
    try:
        import cv2

        T, H, W, C = frames.shape

        # Convert to uint8
        frames_uint8 = (frames * 255).astype(np.uint8)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

        for frame in frames_uint8:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"Saved video to {save_path}")

    except ImportError:
        print("OpenCV not available, cannot create video")


def plot_training_curves(
    losses: List[float],
    accuracies: List[float],
    save_path: str = 'training_curves.png',
):
    """
    Plot training curves

    Args:
        losses: List of loss values
        accuracies: List of accuracy values
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(losses, 'b-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(accuracies, 'r-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {save_path}")


def visualize_4way_classification(
    query_video: torch.Tensor,
    candidate_videos: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    save_path: str = '4way_classification.png',
):
    """
    Visualize 4-way classification results

    Args:
        query_video: Query video of shape (B, T, C, H, W)
        candidate_videos: Candidate videos of shape (B, 4, T, C, H, W)
        predictions: Predicted indices of shape (B,)
        labels: Ground truth indices of shape (B,)
        save_path: Path to save visualization
    """
    B, T, C, H, W = query_video.shape

    # Select first query
    query = query_video[0, 0].cpu().numpy().transpose(1, 2, 0)
    candidates = candidate_videos[0, :, 0].cpu().numpy()
    pred = predictions[0].item()
    label = labels[0].item()

    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # Plot query
    axes[0].imshow(query)
    axes[0].set_title('Query', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot candidates
    for i in range(4):
        candidate = candidates[i].transpose(1, 2, 0)
        axes[i + 1].imshow(candidate)

        title = f'Candidate {i}'
        if i == pred and i == label:
            title += '\n(Pred & GT)'
            color = 'green'
        elif i == pred:
            title += '\n(Predicted)'
            color = 'blue'
        elif i == label:
            title += '\n(Ground Truth)'
            color = 'orange'
        else:
            color = 'black'

        axes[i + 1].set_title(title, fontsize=12, color=color)
        axes[i + 1].axis('off')

        # Add border
        if i == pred:
            for spine in axes[i + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved 4-way classification visualization to {save_path}")
