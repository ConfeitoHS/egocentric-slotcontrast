# Training Quick Start Guide

## Overview

Train an integrated model that combines:
- **Silicon-Menagerie** pretrained ViT (frozen backbone)
- **STEVE** slot attention (trainable object decomposition)
- **SlotContrast** temporal consistency (trainable video understanding)

## Prerequisites

```bash
cd /work/hslee/egocentric-slotcontrast

# Ensure all repositories are present
ls ../steve  # STEVE repository
ls ../silicon-menagerie  # Pretrained models

# Install dependencies
poetry install
```

## Step-by-Step Training

### 1. Prepare Your Data

Update paths in `integrated_model/training/integrated_saycam.yml`:

```yaml
dataset:
  json_dir: "/path/to/saycam/metadata"
  img_dir: "/path/to/saycam/frames"
  batch_size: 16
  ep_length: 8  # frames per video clip
```

### 2. Start Training

```bash
cd integrated_model/training

# Start training
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs
```

### 3. Monitor Progress

```bash
# In another terminal
tensorboard --logdir ./logs

# Open browser to http://localhost:6006
```

**Watch these metrics:**
- `train/loss` - Should decrease steadily
- `train/loss_featrec` - Feature reconstruction (~0.15-0.25)
- `train/loss_ss` - Temporal consistency (~0.3-0.5)

### 4. Checkpoints

Saved every 2500 steps in:
```
logs/integrated_model/silicon_steve_slotcontrast_TIMESTAMP/checkpoints/
```

## Common Adjustments

### Adjust for Your GPU Memory

**If you have OOM errors:**
```bash
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    globals.BATCH_SIZE_PER_GPU=8 \
    dataset.ep_length=6 \
    globals.NUM_SLOTS=5
```

### Adjust Number of Slots

**For simple scenes (few objects):**
```bash
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    globals.NUM_SLOTS=5
```

**For complex scenes (many objects):**
```bash
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    globals.NUM_SLOTS=12
```

### Resume Training

```bash
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    --continue ./logs/experiment_name/checkpoints/step=50000.ckpt
```

### Multi-GPU Training

```bash
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    globals.NUM_GPUS=4 \
    globals.BATCH_SIZE_PER_GPU=16
```

## What's Happening During Training

### Architecture Flow

```
Video Frames (B, 8, 3, 224, 224)
    ↓
Pretrained ViT (Frozen) → Features (B, 8, 256, 768)
    ↓
MLP Transform → Slot Features (B, 8, 256, 128)
    ↓
Slot Attention (Trainable) → Object Slots (B, 8, 7, 128)
    ↓
Temporal Consistency → Consistent Slots Across Time
    ↓
MLP Decoder → Reconstructed Features (B, 8, 256, 768)
```

### Training Objectives

1. **Feature Reconstruction:** Slots should reconstruct ViT features
2. **Temporal Consistency:** Same slot tracks same object across time
3. **Object Decomposition:** Each slot represents one object

### Expected Timeline

**Single GPU (V100/A100):**
- 100K steps: ~18-24 hours
- Loss converges: ~30-50K steps
- Good performance: ~70-100K steps

## After Training

### Extract Features

```python
from integrated_model import SlotFeatureExtractor

# Load trained model
model = SlotFeatureExtractor.load_from_checkpoint(
    'logs/.../checkpoints/step=100000.ckpt'
)

# Extract features
features = model(video)
```

### Evaluate on Downstream Tasks

```bash
cd integrated_model/examples

# Test 4-way classification
python downstream_tasks.py --checkpoint ../training/logs/.../step=100000.ckpt

# Visualize slots
python visualization_demo.py --checkpoint ../training/logs/.../step=100000.ckpt
```

## Troubleshooting

### Training is slow
- Check GPU utilization: `nvidia-smi`
- Increase `dataset.num_workers` (default: 4)
- Enable optimizations: `--use-optimizations`

### Loss is unstable
- Reduce learning rate: `optimizer.lr=0.0002`
- Increase gradient clipping: `trainer.gradient_clip_val=0.1`

### Slots collapse
- Increase attention iterations: `grouper.n_iters=5`
- Reduce temperature: `losses.loss_ss.temperature=0.05`

## Key Configuration Options

```yaml
# Number of object slots
globals.NUM_SLOTS: 7

# Batch size (reduce if OOM)
globals.BATCH_SIZE_PER_GPU: 16

# Video clip length (reduce if OOM)
dataset.ep_length: 8

# Training duration
trainer.max_steps: 100000

# Learning rate
optimizer.lr: 0.0004

# Loss weights
model.loss_weights.loss_featrec: 1.0
model.loss_weights.loss_ss: 0.5
```

## Complete Example

```bash
#!/bin/bash

# Navigate to training directory
cd /work/hslee/egocentric-slotcontrast/integrated_model/training

# Set data paths
DATA_DIR="/path/to/your/saycam/data"
LOG_DIR="./logs"

# Train with custom settings
python train_integrated.py integrated_saycam.yml \
    --data-dir $DATA_DIR \
    --log-dir $LOG_DIR \
    --use-optimizations \
    globals.NUM_SLOTS=7 \
    globals.BATCH_SIZE_PER_GPU=16 \
    dataset.ep_length=8 \
    trainer.max_steps=100000 \
    optimizer.lr=0.0004

# Monitor with tensorboard (in another terminal)
tensorboard --logdir $LOG_DIR
```

## Next Steps

1. **Monitor training** via TensorBoard
2. **Evaluate checkpoints** on downstream tasks
3. **Visualize slots** to verify object decomposition
4. **Fine-tune** for your specific application

For detailed information, see:
- `integrated_model/training/README.md` - Complete training guide
- `integrated_model/README.md` - Model architecture and usage
- `INTEGRATED_MODEL_SUMMARY.md` - Implementation overview
