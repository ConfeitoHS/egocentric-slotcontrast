"""
Downstream Tasks Example

Demonstrates how to use downstream tasks:
1. 4-way classification by similarity score
2. Linear probing
3. Few-shot learning
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slot_feature_extractor import SlotFeatureExtractor
from downstream_tasks import (
    FourWayClassifier,
    LinearProbing,
    FewShotLearner,
    SlotContrastiveLearning
)


def example_4way_classification():
    """Example: 4-way classification by similarity"""
    print("=" * 80)
    print("Example 1: 4-way Classification by Similarity")
    print("=" * 80)

    # Extract features
    feature_extractor = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='concat',
    )

    # Initialize classifier
    fusion_dim = 128 + 768  # slot_size + saycam_feature_dim
    classifier = FourWayClassifier(
        feature_dim=fusion_dim,
        similarity_metric='cosine',
        temperature=0.07,
    )

    # Create dummy data
    query_video = torch.randn(4, 8, 3, 128, 128)  # 4 queries
    candidate_videos = torch.randn(4, 4, 8, 3, 128, 128)  # 4 queries x 4 candidates

    # Extract features
    feature_extractor.eval()
    with torch.no_grad():
        # Extract query features
        query_output = feature_extractor(query_video)
        query_features = query_output['features'].mean(dim=[1, 2])  # (4, fusion_dim)

        # Extract candidate features
        B, N, T, C, H, W = candidate_videos.shape
        candidate_videos_flat = candidate_videos.view(B * N, T, C, H, W)
        candidate_output = feature_extractor(candidate_videos_flat)
        candidate_features = candidate_output['features'].mean(dim=[1, 2])
        candidate_features = candidate_features.view(B, N, -1)  # (4, 4, fusion_dim)

    # Create labels (first candidate is correct)
    labels = torch.zeros(4, dtype=torch.long)

    # Forward pass
    classifier.train()
    output = classifier(query_features, candidate_features, labels)

    print(f"Query features shape: {query_features.shape}")
    print(f"Candidate features shape: {candidate_features.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Predictions: {output['predictions']}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Accuracy: {output['accuracy'].item():.4f}")

    print("\n" + "=" * 80)


def example_linear_probing():
    """Example: Linear probing for classification"""
    print("=" * 80)
    print("Example 2: Linear Probing")
    print("=" * 80)

    # Extract features
    feature_extractor = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='concat',
        freeze_steve=True,
        freeze_saycam=True,
    )

    # Initialize linear probe
    fusion_dim = 128 + 768
    num_classes = 10
    linear_probe = LinearProbing(
        feature_dim=fusion_dim,
        num_classes=num_classes,
        pooling='mean',
        use_slot_attention=False,
    )

    # Create dummy data
    videos = torch.randn(8, 8, 3, 128, 128)  # 8 videos
    labels = torch.randint(0, num_classes, (8,))

    # Extract features (freeze feature extractor)
    feature_extractor.eval()
    with torch.no_grad():
        output = feature_extractor(videos)
        features = output['features']  # (8, T, num_slots, fusion_dim)

    print(f"Extracted features shape: {features.shape}")

    # Train linear probe
    linear_probe.train()
    optimizer = optim.Adam(linear_probe.parameters(), lr=1e-3)

    # Training loop (demo with single batch)
    for epoch in range(3):
        optimizer.zero_grad()

        probe_output = linear_probe(features, labels)
        loss = probe_output['loss']
        accuracy = probe_output['accuracy']

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}, "
              f"Accuracy = {accuracy.item():.4f}")

    print("\n" + "=" * 80)


def example_few_shot_learning():
    """Example: Few-shot learning"""
    print("=" * 80)
    print("Example 3: Few-Shot Learning (5-way 1-shot)")
    print("=" * 80)

    # Extract features
    feature_extractor = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='concat',
    )

    # Initialize few-shot learner
    fusion_dim = 128 + 768
    few_shot_learner = FewShotLearner(
        feature_dim=fusion_dim,
        distance_metric='euclidean',
    )

    # Create dummy 5-way 1-shot task
    num_classes = 5
    n_support = 1  # 1-shot
    n_query = 5  # 5 queries per class

    # Support set: 5 classes x 1 example
    support_videos = torch.randn(num_classes * n_support, 8, 3, 128, 128)
    support_labels = torch.arange(num_classes).repeat_interleave(n_support)

    # Query set: 5 classes x 5 examples
    query_videos = torch.randn(num_classes * n_query, 8, 3, 128, 128)
    query_labels = torch.arange(num_classes).repeat_interleave(n_query)

    # Extract features
    feature_extractor.eval()
    with torch.no_grad():
        support_output = feature_extractor(support_videos)
        support_features = support_output['features'].mean(dim=[1, 2])

        query_output = feature_extractor(query_videos)
        query_features = query_output['features'].mean(dim=[1, 2])

    print(f"Support features shape: {support_features.shape}")
    print(f"Query features shape: {query_features.shape}")

    # Few-shot classification
    few_shot_learner.eval()
    with torch.no_grad():
        output = few_shot_learner(
            support_features,
            support_labels,
            query_features,
            query_labels,
            num_classes=num_classes,
        )

    print(f"Prototypes shape: {output['prototypes'].shape}")
    print(f"Predictions: {output['predictions']}")
    print(f"Accuracy: {output['accuracy'].item():.4f}")

    print("\n" + "=" * 80)


def example_contrastive_learning():
    """Example: Contrastive learning on slots"""
    print("=" * 80)
    print("Example 4: Contrastive Learning on Slots")
    print("=" * 80)

    # Extract features
    feature_extractor = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='concat',
    )

    # Initialize contrastive learner
    fusion_dim = 128 + 768
    contrastive_learner = SlotContrastiveLearning(
        feature_dim=fusion_dim,
        projection_dim=128,
        temperature=0.07,
    )

    # Create augmented views of same videos
    videos_1 = torch.randn(8, 8, 3, 128, 128)
    videos_2 = torch.randn(8, 8, 3, 128, 128)  # Different augmentation

    # Extract features
    feature_extractor.eval()
    with torch.no_grad():
        output_1 = feature_extractor(videos_1)
        features_1 = output_1['features'].mean(dim=[1, 2])  # (8, fusion_dim)

        output_2 = feature_extractor(videos_2)
        features_2 = output_2['features'].mean(dim=[1, 2])  # (8, fusion_dim)

    # Compute contrastive loss
    contrastive_learner.train()
    output = contrastive_learner(features_1, features_2)

    print(f"Features 1 shape: {features_1.shape}")
    print(f"Features 2 shape: {features_2.shape}")
    print(f"Contrastive loss: {output['loss'].item():.4f}")
    print(f"Logits shape: {output['logits'].shape}")

    print("\n" + "=" * 80)


def example_full_pipeline():
    """Example: Complete training pipeline"""
    print("=" * 80)
    print("Example 5: Complete Training Pipeline")
    print("=" * 80)

    print("This example demonstrates a complete training pipeline:")
    print("1. Extract features with frozen pretrained models")
    print("2. Train downstream classifier")
    print("3. Evaluate on test set")
    print()

    # Initialize models
    feature_extractor = SlotFeatureExtractor(
        saycam_model_name='dino_say_vitb14',
        num_slots=7,
        slot_size=128,
        feature_fusion='concat',
        freeze_steve=True,
        freeze_saycam=True,
    )

    fusion_dim = 128 + 768
    num_classes = 5
    classifier = LinearProbing(
        feature_dim=fusion_dim,
        num_classes=num_classes,
        pooling='attention',
        num_slots=7,
        use_slot_attention=True,
    )

    # Create dummy train/test data
    train_videos = torch.randn(20, 8, 3, 128, 128)
    train_labels = torch.randint(0, num_classes, (20,))

    test_videos = torch.randn(10, 8, 3, 128, 128)
    test_labels = torch.randint(0, num_classes, (10,))

    # Training
    print("Training...")
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    num_epochs = 5
    batch_size = 4

    feature_extractor.eval()
    classifier.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for i in range(0, len(train_videos), batch_size):
            batch_videos = train_videos[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Extract features
            with torch.no_grad():
                output = feature_extractor(batch_videos)
                features = output['features']

            # Train classifier
            optimizer.zero_grad()
            classifier_output = classifier(features, batch_labels)
            loss = classifier_output['loss']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += classifier_output['accuracy'].item()
            num_batches += 1

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Loss = {epoch_loss / num_batches:.4f}, "
              f"Accuracy = {epoch_acc / num_batches:.4f}")

    # Evaluation
    print("\nEvaluating on test set...")
    classifier.eval()

    with torch.no_grad():
        test_output = feature_extractor(test_videos)
        test_features = test_output['features']
        test_classifier_output = classifier(test_features, test_labels)

    print(f"Test Loss: {test_classifier_output['loss'].item():.4f}")
    print(f"Test Accuracy: {test_classifier_output['accuracy'].item():.4f}")

    print("\n" + "=" * 80)


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("DOWNSTREAM TASKS EXAMPLES")
    print("=" * 80 + "\n")

    example_4way_classification()
    example_linear_probing()
    example_few_shot_learning()
    example_contrastive_learning()
    example_full_pipeline()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
