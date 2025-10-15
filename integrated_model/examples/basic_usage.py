"""
Basic Usage Example

Demonstrates how to use the integrated slot feature extractor
on egocentric videos
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slot_feature_extractor import SlotFeatureExtractor
from data_utils import load_egocentric_video, EgocentricVideoDataset, create_dataloader


def example_single_video():
    """Example: Extract features from a single video"""
    print("=" * 80)
    print("Example 1: Extract features from a single video")
    print("=" * 80)

    # Initialize the model
    model = SlotFeatureExtractor(
        steve_checkpoint_path=None,  # Use randomly initialized STEVE
        saycam_model_name='dino_say_vitb14',  # Pretrained SAYCam model
        num_slots=7,
        slot_size=128,
        freeze_steve=False,  # Allow finetuning
        freeze_saycam=True,  # Keep pretrained features frozen
        feature_fusion='concat',
    )

    # Load a video (replace with actual video path)
    video_path = 'path/to/your/egocentric_video.mp4'

    # Check if video exists, otherwise create dummy data
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}, using dummy data...")
        video = torch.randn(1, 8, 3, 128, 128)  # (B, T, C, H, W)
    else:
        video = load_egocentric_video(
            video_path,
            num_frames=8,
            frame_size=(128, 128),
            normalize=True,
        )

    print(f"Video shape: {video.shape}")

    # Extract features
    model.eval()
    with torch.no_grad():
        output = model(video, return_attention=True)

    print(f"\nOutput keys: {output.keys()}")
    print(f"Features shape: {output['features'].shape}")
    print(f"Slots shape: {output['slots'].shape}")
    print(f"Visual features shape: {output['visual_features'].shape}")

    if 'attention_vis' in output:
        print(f"Attention visualization shape: {output['attention_vis'].shape}")

    # Get aggregated representation
    aggregated = model.get_object_centric_representation(video, aggregate='mean')
    print(f"\nAggregated representation shape: {aggregated.shape}")

    print("\n" + "=" * 80)


def example_batch_processing():
    """Example: Process a batch of videos"""
    print("=" * 80)
    print("Example 2: Process a batch of videos")
    print("=" * 80)

    # Initialize model
    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='attention',  # Use attention-based fusion
    )

    # Create dummy dataset (replace with actual video paths)
    video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4']
    labels = [0, 1, 0, 1]

    # Create dummy data since videos don't exist
    print("Using dummy data for demonstration...")
    dummy_videos = torch.randn(4, 8, 3, 128, 128)  # (B, T, C, H, W)

    # Process batch
    model.eval()
    with torch.no_grad():
        output = model(dummy_videos, return_attention=False)

    print(f"Batch features shape: {output['features'].shape}")
    print(f"Batch slots shape: {output['slots'].shape}")

    print("\n" + "=" * 80)


def example_with_dataloader():
    """Example: Use with PyTorch DataLoader"""
    print("=" * 80)
    print("Example 3: Using with PyTorch DataLoader")
    print("=" * 80)

    # Initialize model
    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
    )

    # Create dummy dataset
    print("Creating dummy dataset...")
    dummy_video_dir = '/path/to/videos'  # Replace with actual directory

    # Since we don't have real videos, we'll create a custom dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                'video': torch.randn(8, 3, 128, 128),
                'label': torch.tensor(idx % 2),
            }

    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    # Process batches
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            video = batch['video']
            labels = batch['label']

            output = model(video)
            features = output['features']

            all_features.append(features)

            print(f"Batch {batch_idx}: video shape {video.shape}, "
                  f"features shape {features.shape}")

            if batch_idx >= 2:  # Process only first 3 batches for demo
                break

    print("\n" + "=" * 80)


def example_extract_slots_only():
    """Example: Extract only slot representations"""
    print("=" * 80)
    print("Example 4: Extract only slot representations")
    print("=" * 80)

    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
    )

    # Create dummy video
    video = torch.randn(2, 8, 3, 128, 128)

    # Extract only slots (faster, doesn't use SAYCam features)
    model.eval()
    with torch.no_grad():
        slots = model.extract_slot_features_only(video)

    print(f"Extracted slots shape: {slots.shape}")
    print("This is faster as it only uses STEVE encoder")

    print("\n" + "=" * 80)


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("INTEGRATED SLOT FEATURE EXTRACTOR - BASIC USAGE EXAMPLES")
    print("=" * 80 + "\n")

    example_single_video()
    example_batch_processing()
    example_with_dataloader()
    example_extract_slots_only()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
