# Integrated Object-Centric Feature Extractor for Egocentric Videos

An integrated model that combines **STEVE** (Slot-Transformer for Videos) and **Silicon-Menagerie** (pretrained transformers on SAYCam dataset) to extract object-centric features from egocentric videos. This is built on top of the **SlotContrast** framework.

## Overview

This integrated model addresses the challenge of extracting object-centric representations from egocentric videos by combining:

1. **STEVE**: Generates slots representing objects/background in video frames
2. **SAYCam Transformer**: Provides rich pretrained visual features from child's view egocentric videos
3. **Feature Fusion**: Combines slot and visual features for downstream tasks

### Architecture

```
Egocentric Video (B, T, C, H, W)
          |
          ├─────> STEVE Encoder ─────> Slots (B, T, num_slots, slot_size)
          |                              |
          |                              v
          |                       Temporal Encoding
          |                              |
          └─────> SAYCam Transformer ──> Visual Features (B, T, feature_dim)
                                         |
                                         v
                                   Feature Fusion
                                         |
                                         v
                            Object-Centric Features (B, T, num_slots, fusion_dim)
                                         |
                                         v
                                  Downstream Tasks
                            (4-way classification, linear probing, etc.)
```

## Installation

### Prerequisites

```bash
cd /work/hslee
```

Make sure you have the following repositories:
- `egocentric-slotcontrast/` (this repository)
- `steve/` (STEVE model)
- `silicon-menagerie/` (SAYCam pretrained models)

### Dependencies

```bash
cd egocentric-slotcontrast
poetry install

# Install additional dependencies
pip install torchvision opencv-python huggingface-hub
```

## Quick Start

### Basic Usage

```python
import torch
from integrated_model import SlotFeatureExtractor, load_egocentric_video

# Initialize the model
model = SlotFeatureExtractor(
    steve_checkpoint_path=None,  # Optional: path to pretrained STEVE
    saycam_model_name='dino_say_vitb14',  # Pretrained SAYCam model
    num_slots=7,  # Number of object slots
    slot_size=128,  # Slot dimension
    freeze_steve=True,  # Freeze STEVE weights
    freeze_saycam=True,  # Freeze SAYCam weights
    feature_fusion='concat',  # 'concat', 'add', or 'attention'
)

# Load video
video = load_egocentric_video(
    'path/to/video.mp4',
    num_frames=8,
    frame_size=(128, 128),
)

# Extract features
model.eval()
with torch.no_grad():
    output = model(video, return_attention=True)

# Access results
features = output['features']  # (B, T, num_slots, fusion_dim)
slots = output['slots']  # (B, T, num_slots, slot_size)
visual_features = output['visual_features']  # (B, T, feature_dim)
attention_maps = output['attention_maps']  # (B, T, num_slots, 1, H, W)
```

## Components

### 1. Slot Feature Extractor

Main model that combines STEVE and SAYCam transformers.

**Key Parameters:**
- `num_slots`: Number of slots for object decomposition (default: 7)
- `slot_size`: Dimensionality of each slot (default: 128)
- `saycam_model_name`: Pretrained SAYCam model (e.g., 'dino_say_vitb14')
- `feature_fusion`: How to combine features ('concat', 'add', 'attention')
- `freeze_steve`, `freeze_saycam`: Whether to freeze pretrained weights

**Available SAYCam Models:**
- `dino_say_vitb14`: DINO ViT-B/14 trained on full SAYCam
- `dino_s_vitb14`: Trained on baby S
- `dino_a_vitb14`: Trained on baby A
- `dino_y_vitb14`: Trained on baby Y
- `mae_say_vitb14`: MAE ViT-B/14 trained on SAYCam
- Many more (see silicon-menagerie docs)

### 2. Downstream Tasks

#### 4-Way Classification

Classify by comparing query with 4 candidates based on similarity.

```python
from integrated_model import FourWayClassifier

classifier = FourWayClassifier(
    feature_dim=896,  # slot_size + saycam_feature_dim
    similarity_metric='cosine',
    temperature=0.07,
)

output = classifier(query_features, candidate_features, labels)
```

#### Linear Probing

Train linear classifier on frozen features.

```python
from integrated_model import LinearProbing

probe = LinearProbing(
    feature_dim=896,
    num_classes=10,
    pooling='mean',  # 'mean', 'max', 'attention'
    use_slot_attention=False,
)

output = probe(features, labels)
```

#### Few-Shot Learning

Prototypical networks for few-shot classification.

```python
from integrated_model import FewShotLearner

learner = FewShotLearner(
    feature_dim=896,
    distance_metric='euclidean',
)

output = learner(
    support_features, support_labels,
    query_features, query_labels,
    num_classes=5
)
```

### 3. Data Loading

```python
from integrated_model import EgocentricVideoDataset, create_dataloader

# Create dataset
dataset = EgocentricVideoDataset(
    video_paths='path/to/videos',  # Directory or list of paths
    labels=[0, 1, 2, ...],  # Optional labels
    num_frames=8,
    frame_size=(128, 128),
    temporal_sampling='uniform',  # 'uniform', 'random', 'consecutive'
)

# Create dataloader
dataloader = create_dataloader(
    video_paths='path/to/videos',
    batch_size=8,
    num_frames=8,
    shuffle=True,
    num_workers=4,
)
```

## Examples

We provide comprehensive examples in the `examples/` directory:

### 1. Basic Usage (`examples/basic_usage.py`)

```bash
cd integrated_model/examples
python basic_usage.py
```

Demonstrates:
- Single video processing
- Batch processing
- Using with DataLoader
- Extracting only slots

### 2. Downstream Tasks (`examples/downstream_tasks.py`)

```bash
cd integrated_model/examples
python downstream_tasks.py
```

Demonstrates:
- 4-way classification
- Linear probing
- Few-shot learning
- Contrastive learning
- Complete training pipeline

## Feature Fusion Strategies

### 1. Concatenation (`feature_fusion='concat'`)
Simple concatenation of slot and visual features.
- Pros: Preserves all information
- Cons: Increases dimensionality

### 2. Addition (`feature_fusion='add'`)
Projects visual features to slot dimension and adds.
- Pros: Maintains slot dimension
- Cons: May lose information

### 3. Attention (`feature_fusion='attention'`)
Cross-attention between slots and visual features.
- Pros: Learned interaction
- Cons: More parameters, slower

## Training Strategies

### Strategy 1: Frozen Features + Linear Probe

Best for quick evaluation and when data is limited.

```python
model = SlotFeatureExtractor(
    freeze_steve=True,
    freeze_saycam=True,
)

probe = LinearProbing(feature_dim=896, num_classes=10)

# Train only the linear probe
optimizer = optim.Adam(probe.parameters(), lr=1e-3)
```

### Strategy 2: Fine-tune STEVE, Freeze SAYCam

When you want to adapt slot representations to your data.

```python
model = SlotFeatureExtractor(
    freeze_steve=False,  # Allow finetuning
    freeze_saycam=True,  # Keep pretrained features
)

# Train both feature extractor and downstream task
optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 1e-4},
    {'params': downstream_task.parameters(), 'lr': 1e-3}
])
```

### Strategy 3: End-to-End Fine-tuning

When you have sufficient data.

```python
model = SlotFeatureExtractor(
    freeze_steve=False,
    freeze_saycam=False,
)

# Train everything
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

## Downstream Task Applications

### 1. Object Recognition
Use linear probing to classify objects in egocentric videos.

### 2. Action Recognition
Pool slot features across time for action classification.

### 3. Video Retrieval
Use 4-way classification for video matching.

### 4. Few-Shot Learning
Learn from few examples using prototypical networks.

### 5. Object Tracking
Use temporal consistency of slots to track objects.

## Performance Considerations

### Memory Usage

- **Slots**: Increase with `num_slots` and `slot_size`
- **SAYCam features**: Fixed by model choice
- **Fusion**: Concatenation uses most memory

**Tips:**
- Use `feature_fusion='add'` to reduce memory
- Process fewer frames at once
- Use gradient checkpointing for large models

### Speed

- **STEVE**: Moderate (slot attention iterations)
- **SAYCam**: Fast (single forward pass)
- **Fusion**: Fast (except attention)

**Tips:**
- Use `extract_slot_features_only()` if you don't need SAYCam
- Batch videos together
- Use smaller SAYCam models (e.g., ViT-S instead of ViT-B)

## Advanced Usage

### Custom Feature Fusion

You can extend the `SlotFeatureExtractor` class to implement custom fusion:

```python
class CustomSlotFeatureExtractor(SlotFeatureExtractor):
    def fuse_features(self, slots, visual_features):
        # Your custom fusion logic
        pass
```

### Custom Downstream Tasks

Implement your own downstream tasks by extending base classes:

```python
class MyCustomTask(nn.Module):
    def forward(self, features, labels=None):
        # Your task logic
        pass
```

## Troubleshooting

### Issue: Out of Memory

**Solution:**
- Reduce batch size
- Reduce number of frames
- Use smaller models
- Use `feature_fusion='add'` instead of `'concat'`

### Issue: STEVE/SAYCam imports fail

**Solution:**
Ensure repositories are in correct locations:
```
/work/hslee/
├── egocentric-slotcontrast/
├── steve/
└── silicon-menagerie/
```

### Issue: Video loading fails

**Solution:**
- Install opencv: `pip install opencv-python`
- Or use torchvision: `pip install torchvision`

## Citation

If you use this integrated model, please cite the original works:

```bibtex
@inproceedings{manasyan2025temporally,
    title={Temporally Consistent Object-Centric Learning by Contrasting Slots},
    author={Manasyan, Anna and Seitzer, Maximilian and Radovic, Filip and Martius, Georg and Zadaianchuk, Andrii},
    booktitle={CVPR},
    year={2025}
}

@inproceedings{singh2022simple,
    title={Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos},
    author={Singh, Gautam and Wu, Yi-Fu and Ahn, Sungjin},
    booktitle={NeurIPS},
    year={2022}
}

@article{orhan2023learning,
    title={Learning high-level visual representations from a child's perspective without strong inductive biases},
    author={Orhan, AE and Lake, BM},
    journal={arXiv preprint arXiv:2305.15372},
    year={2023}
}
```

## License

This integrated model inherits licenses from:
- SlotContrast: MIT License
- STEVE: See steve/LICENSE
- Silicon-Menagerie: See silicon-menagerie/LICENSE

## Contact

For questions or issues, please open an issue in the egocentric-slotcontrast repository.
