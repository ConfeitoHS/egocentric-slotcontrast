"""
Visualization Demo

Demonstrates various visualization capabilities
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slot_feature_extractor import SlotFeatureExtractor
from visualization_utils import (
    visualize_slots,
    visualize_attention_evolution,
    visualize_slot_features,
    visualize_feature_similarity,
    plot_training_curves,
    visualize_4way_classification,
)


def demo_slot_visualization():
    """Demo: Visualize slots and attention maps"""
    print("=" * 80)
    print("Demo 1: Slot and Attention Visualization")
    print("=" * 80)

    # Initialize model
    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
    )

    # Create dummy video
    video = torch.randn(2, 8, 3, 128, 128)

    # Extract features with attention
    model.eval()
    with torch.no_grad():
        output = model(video, return_attention=True)

    # Visualize
    visualize_slots(
        video=video,
        slots=output['slots'],
        attention_maps=output['attention_maps'],
        save_path='demo_slots.png',
        max_frames=8,
        max_slots=7,
    )

    print("\n" + "=" * 80)


def demo_attention_evolution():
    """Demo: Visualize attention evolution over time"""
    print("=" * 80)
    print("Demo 2: Attention Evolution Over Time")
    print("=" * 80)

    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
    )

    video = torch.randn(1, 8, 3, 128, 128)

    model.eval()
    with torch.no_grad():
        output = model(video, return_attention=True)

    # Visualize evolution of first slot
    visualize_attention_evolution(
        attention_maps=output['attention_maps'],
        save_path='demo_attention_evolution.png',
        slot_idx=0,
    )

    print("\n" + "=" * 80)


def demo_feature_visualization():
    """Demo: Visualize slot features in 2D"""
    print("=" * 80)
    print("Demo 3: Slot Feature Visualization (PCA)")
    print("=" * 80)

    try:
        model = SlotFeatureExtractor(
            saycam_model_name='dino_say_vitb14',
            num_slots=7,
            slot_size=128,
        )

        video = torch.randn(2, 8, 3, 128, 128)

        model.eval()
        with torch.no_grad():
            output = model(video, return_attention=False)

        # Visualize slot features
        visualize_slot_features(
            slots=output['slots'],
            save_path='demo_slot_features_pca.png',
            method='pca',
        )

        print("PCA visualization created successfully")

    except ImportError as e:
        print(f"Skipping: {e}")
        print("Install scikit-learn: pip install scikit-learn")

    print("\n" + "=" * 80)


def demo_similarity_matrix():
    """Demo: Visualize feature similarity matrix"""
    print("=" * 80)
    print("Demo 4: Feature Similarity Matrix")
    print("=" * 80)

    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
    )

    # Create videos from 3 classes
    videos = torch.randn(12, 8, 3, 128, 128)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    # Extract features
    model.eval()
    with torch.no_grad():
        output = model(videos)
        features = output['features'].mean(dim=[1, 2])  # (12, fusion_dim)

    # Visualize similarity
    visualize_feature_similarity(
        features=features,
        labels=labels,
        save_path='demo_similarity_matrix.png',
    )

    print("\n" + "=" * 80)


def demo_training_curves():
    """Demo: Plot training curves"""
    print("=" * 80)
    print("Demo 5: Training Curves")
    print("=" * 80)

    # Simulate training data
    import numpy as np

    num_iters = 100
    losses = []
    accuracies = []

    for i in range(num_iters):
        # Simulated decreasing loss
        loss = 2.0 * np.exp(-i / 20) + 0.1 * np.random.rand()
        losses.append(loss)

        # Simulated increasing accuracy
        acc = 1.0 - 0.9 * np.exp(-i / 15) + 0.05 * np.random.rand()
        accuracies.append(acc)

    plot_training_curves(
        losses=losses,
        accuracies=accuracies,
        save_path='demo_training_curves.png',
    )

    print("\n" + "=" * 80)


def demo_4way_visualization():
    """Demo: Visualize 4-way classification"""
    print("=" * 80)
    print("Demo 6: 4-Way Classification Visualization")
    print("=" * 80)

    # Create dummy query and candidates
    query = torch.randn(2, 8, 3, 128, 128)
    candidates = torch.randn(2, 4, 8, 3, 128, 128)

    # Simulate predictions and labels
    predictions = torch.tensor([1, 0])  # Predicted indices
    labels = torch.tensor([1, 0])  # Ground truth indices

    visualize_4way_classification(
        query_video=query,
        candidate_videos=candidates,
        predictions=predictions,
        labels=labels,
        save_path='demo_4way_classification.png',
    )

    print("\n" + "=" * 80)


def demo_comprehensive_visualization():
    """Demo: Comprehensive visualization of all outputs"""
    print("=" * 80)
    print("Demo 7: Comprehensive Visualization")
    print("=" * 80)

    model = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='concat',
    )

    # Create video
    video = torch.randn(2, 8, 3, 128, 128)

    # Extract all features
    model.eval()
    with torch.no_grad():
        output = model(video, return_attention=True)

    print("Output keys:", output.keys())
    print(f"Features shape: {output['features'].shape}")
    print(f"Slots shape: {output['slots'].shape}")
    print(f"Visual features shape: {output['visual_features'].shape}")
    print(f"Attention maps shape: {output['attention_maps'].shape}")
    print(f"Attention vis shape: {output['attention_vis'].shape}")

    # Create multiple visualizations
    print("\nCreating visualizations...")

    # 1. Slots visualization
    visualize_slots(
        video=video,
        slots=output['slots'],
        attention_maps=output['attention_maps'],
        save_path='demo_comprehensive_slots.png',
    )

    # 2. Attention evolution for multiple slots
    for slot_idx in [0, 2, 4]:
        visualize_attention_evolution(
            attention_maps=output['attention_maps'],
            save_path=f'demo_comprehensive_attn_slot{slot_idx}.png',
            slot_idx=slot_idx,
        )

    # 3. Feature visualization
    try:
        visualize_slot_features(
            slots=output['slots'],
            save_path='demo_comprehensive_features.png',
            method='pca',
        )
    except ImportError:
        print("Skipping feature visualization (scikit-learn not available)")

    print("\nAll visualizations created successfully!")
    print("\n" + "=" * 80)


def main():
    """Run all visualization demos"""
    print("\n" + "=" * 80)
    print("VISUALIZATION DEMOS")
    print("=" * 80 + "\n")

    # Create output directory if it doesn't exist
    os.makedirs('visualization_outputs', exist_ok=True)
    os.chdir('visualization_outputs')

    try:
        demo_slot_visualization()
        demo_attention_evolution()
        demo_feature_visualization()
        demo_similarity_matrix()
        demo_training_curves()
        demo_4way_visualization()
        demo_comprehensive_visualization()

        print("\n" + "=" * 80)
        print("All demos completed!")
        print(f"Visualizations saved in: {os.getcwd()}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
