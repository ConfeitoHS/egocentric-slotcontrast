"""
Downstream Task Interfaces

Implements various downstream tasks for evaluating object-centric features:
1. 4-way classification by similarity score
2. Linear probing of object representations
3. Few-shot learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FourWayClassifier(nn.Module):
    """
    4-way classification by similarity score

    Given a query and 4 candidate slots/features, predict which candidate
    matches the query based on similarity scores.
    """

    def __init__(
        self,
        feature_dim: int,
        similarity_metric: str = 'cosine',  # 'cosine', 'dot', 'l2'
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            similarity_metric: Type of similarity metric to use
            temperature: Temperature for softmax (if using cosine similarity)
            learnable_temperature: Whether to make temperature learnable
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.similarity_metric = similarity_metric

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        # Optional projection layers for learned similarity
        self.query_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.candidate_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def compute_similarity(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between query and candidates

        Args:
            query: Query features of shape (B, feature_dim)
            candidates: Candidate features of shape (B, 4, feature_dim)

        Returns:
            similarities: Similarity scores of shape (B, 4)
        """
        # Project features
        query_proj = self.query_projection(query)  # (B, feature_dim)
        candidates_proj = self.candidate_projection(candidates)  # (B, 4, feature_dim)

        if self.similarity_metric == 'cosine':
            # Normalize features
            query_norm = F.normalize(query_proj, p=2, dim=-1)
            candidates_norm = F.normalize(candidates_proj, p=2, dim=-1)

            # Compute cosine similarity
            similarities = torch.einsum('bd,bnd->bn', query_norm, candidates_norm)
            similarities = similarities / self.temperature

        elif self.similarity_metric == 'dot':
            # Dot product similarity
            similarities = torch.einsum('bd,bnd->bn', query_proj, candidates_proj)

        elif self.similarity_metric == 'l2':
            # Negative L2 distance
            query_expanded = query_proj.unsqueeze(1)  # (B, 1, feature_dim)
            distances = torch.norm(candidates_proj - query_expanded, p=2, dim=-1)
            similarities = -distances  # (B, 4)

        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return similarities

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 4-way classification

        Args:
            query: Query features of shape (B, feature_dim)
            candidates: Candidate features of shape (B, 4, feature_dim)
            labels: Ground truth labels of shape (B,) with values in [0, 1, 2, 3]

        Returns:
            Dictionary containing:
                - logits: Similarity scores (B, 4)
                - predictions: Predicted class indices (B,)
                - loss: Cross-entropy loss (if labels provided)
                - accuracy: Classification accuracy (if labels provided)
        """
        # Compute similarities
        similarities = self.compute_similarity(query, candidates)

        # Get predictions
        predictions = similarities.argmax(dim=-1)

        output = {
            'logits': similarities,
            'predictions': predictions,
        }

        # Compute loss and accuracy if labels provided
        if labels is not None:
            loss = F.cross_entropy(similarities, labels)
            accuracy = (predictions == labels).float().mean()

            output['loss'] = loss
            output['accuracy'] = accuracy

        return output


class LinearProbing(nn.Module):
    """
    Linear probing for object classification

    Trains a linear classifier on top of frozen object-centric features
    to evaluate the quality of learned representations.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        pooling: str = 'mean',  # 'mean', 'max', 'attention', 'cls'
        num_slots: Optional[int] = None,
        use_slot_attention: bool = False,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            num_classes: Number of output classes
            pooling: How to pool slot features
            num_slots: Number of slots (needed for attention pooling)
            use_slot_attention: Whether to use learnable attention over slots
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.num_slots = num_slots
        self.use_slot_attention = use_slot_attention

        # Learnable attention weights for pooling
        if use_slot_attention and num_slots is not None:
            self.slot_attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
            )

        # Linear classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool features across slots and/or time

        Args:
            features: Input features of shape (B, T, num_slots, feature_dim)
                     or (B, num_slots, feature_dim) or (B, T, feature_dim)

        Returns:
            pooled: Pooled features of shape (B, feature_dim)
        """
        if features.dim() == 4:
            # (B, T, num_slots, feature_dim)
            if self.pooling == 'mean':
                pooled = features.mean(dim=[1, 2])
            elif self.pooling == 'max':
                pooled = features.flatten(1, 2).max(dim=1)[0]
            elif self.pooling == 'attention' and self.use_slot_attention:
                # Compute attention weights
                B, T, num_slots, D = features.shape
                features_flat = features.view(B * T * num_slots, D)
                attn_logits = self.slot_attention(features_flat)
                attn_weights = attn_logits.view(B, T * num_slots, 1)
                attn_weights = F.softmax(attn_weights, dim=1)

                # Apply attention
                features_flat = features.view(B, T * num_slots, D)
                pooled = (features_flat * attn_weights).sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")

        elif features.dim() == 3:
            # (B, num_slots, feature_dim) or (B, T, feature_dim)
            if self.pooling == 'mean':
                pooled = features.mean(dim=1)
            elif self.pooling == 'max':
                pooled = features.max(dim=1)[0]
            elif self.pooling == 'attention' and self.use_slot_attention:
                # Compute attention weights
                B, N, D = features.shape
                features_flat = features.view(B * N, D)
                attn_logits = self.slot_attention(features_flat)
                attn_weights = attn_logits.view(B, N, 1)
                attn_weights = F.softmax(attn_weights, dim=1)

                # Apply attention
                pooled = (features * attn_weights).sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")

        elif features.dim() == 2:
            # (B, feature_dim) - already pooled
            pooled = features

        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        return pooled

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for linear probing

        Args:
            features: Input features (various shapes supported)
            labels: Ground truth labels of shape (B,)

        Returns:
            Dictionary containing:
                - logits: Classification logits (B, num_classes)
                - predictions: Predicted class indices (B,)
                - loss: Cross-entropy loss (if labels provided)
                - accuracy: Classification accuracy (if labels provided)
        """
        # Pool features
        pooled_features = self.pool_features(features)

        # Classify
        logits = self.classifier(pooled_features)
        predictions = logits.argmax(dim=-1)

        output = {
            'logits': logits,
            'predictions': predictions,
        }

        # Compute loss and accuracy if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            accuracy = (predictions == labels).float().mean()

            output['loss'] = loss
            output['accuracy'] = accuracy

        return output


class FewShotLearner(nn.Module):
    """
    Few-shot learning with object-centric features

    Uses metric learning (prototypical networks) for few-shot classification
    """

    def __init__(
        self,
        feature_dim: int,
        distance_metric: str = 'euclidean',  # 'euclidean', 'cosine'
    ):
        """
        Args:
            feature_dim: Dimension of input features
            distance_metric: Distance metric for prototype matching
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.distance_metric = distance_metric

        # Optional feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set

        Args:
            support_features: Features of shape (N_support, feature_dim)
            support_labels: Labels of shape (N_support,)
            num_classes: Number of classes

        Returns:
            prototypes: Class prototypes of shape (num_classes, feature_dim)
        """
        prototypes = []
        for c in range(num_classes):
            mask = support_labels == c
            if mask.sum() > 0:
                class_features = support_features[mask]
                prototype = class_features.mean(dim=0)
            else:
                prototype = torch.zeros(self.feature_dim, device=support_features.device)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes, dim=0)
        return prototypes

    def compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between queries and prototypes

        Args:
            query_features: Features of shape (N_query, feature_dim)
            prototypes: Prototypes of shape (num_classes, feature_dim)

        Returns:
            distances: Distance matrix of shape (N_query, num_classes)
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            query_expanded = query_features.unsqueeze(1)  # (N_query, 1, feature_dim)
            prototypes_expanded = prototypes.unsqueeze(0)  # (1, num_classes, feature_dim)
            distances = torch.norm(query_expanded - prototypes_expanded, p=2, dim=-1)

        elif self.distance_metric == 'cosine':
            # Cosine distance
            query_norm = F.normalize(query_features, p=2, dim=-1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
            similarities = torch.mm(query_norm, prototypes_norm.t())
            distances = 1 - similarities

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for few-shot learning

        Args:
            support_features: Support set features (N_support, feature_dim)
            support_labels: Support set labels (N_support,)
            query_features: Query set features (N_query, feature_dim)
            query_labels: Query set labels (N_query,) - optional
            num_classes: Number of classes (inferred from labels if not provided)

        Returns:
            Dictionary containing:
                - distances: Distance matrix (N_query, num_classes)
                - predictions: Predicted class indices (N_query,)
                - loss: Cross-entropy loss (if query_labels provided)
                - accuracy: Classification accuracy (if query_labels provided)
        """
        # Transform features
        support_features = self.feature_transform(support_features)
        query_features = self.feature_transform(query_features)

        # Infer number of classes if not provided
        if num_classes is None:
            num_classes = support_labels.max().item() + 1

        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels, num_classes)

        # Compute distances
        distances = self.compute_distances(query_features, prototypes)

        # Get predictions (closest prototype)
        predictions = distances.argmin(dim=-1)

        output = {
            'distances': distances,
            'predictions': predictions,
            'prototypes': prototypes,
        }

        # Compute loss and accuracy if labels provided
        if query_labels is not None:
            # Convert distances to logits (negative distances with temperature)
            logits = -distances
            loss = F.cross_entropy(logits, query_labels)
            accuracy = (predictions == query_labels).float().mean()

            output['loss'] = loss
            output['accuracy'] = accuracy

        return output


class SlotContrastiveLearning(nn.Module):
    """
    Contrastive learning on slot representations

    Learns better slot features through contrastive objectives
    """

    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            projection_dim: Dimension of projection head
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.temperature = temperature

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss between two sets of features

        Args:
            features_1: First set of features (B, feature_dim)
            features_2: Second set of features (B, feature_dim)

        Returns:
            Dictionary containing:
                - loss: Contrastive loss
                - logits: Similarity matrix
        """
        # Project features
        z1 = self.projection_head(features_1)
        z2 = self.projection_head(features_2)

        # Normalize
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        # Compute similarity matrix
        logits_11 = torch.mm(z1, z1.t()) / self.temperature
        logits_22 = torch.mm(z2, z2.t()) / self.temperature
        logits_12 = torch.mm(z1, z2.t()) / self.temperature
        logits_21 = torch.mm(z2, z1.t()) / self.temperature

        # Create labels (diagonal elements are positives)
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=z1.device)

        # Compute contrastive loss
        loss_1 = F.cross_entropy(logits_12, labels)
        loss_2 = F.cross_entropy(logits_21, labels)
        loss = (loss_1 + loss_2) / 2

        return {
            'loss': loss,
            'logits': logits_12,
        }
