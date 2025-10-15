"""
Integrated Object-Centric Feature Extractor

This module integrates:
1. STEVE: Slot-Transformer for Videos (slot generation from videos)
2. Silicon-Menagerie: Pretrained transformers on SAYCam dataset
3. SlotContrast: Object-centric learning framework

The integrated model extracts object-centric features from egocentric videos
where each slot represents an object or background in the scene.
"""

from .slot_feature_extractor import SlotFeatureExtractor
from .downstream_tasks import FourWayClassifier, LinearProbing
from .data_utils import EgocentricVideoDataset, load_egocentric_video

__all__ = [
    'SlotFeatureExtractor',
    'FourWayClassifier',
    'LinearProbing',
    'EgocentricVideoDataset',
    'load_egocentric_video',
]
