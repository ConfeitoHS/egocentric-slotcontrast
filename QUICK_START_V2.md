# Quick Start Guide - Integrated Model V2

## What This Is

A unified model combining:
- **SlotContrast** (temporal consistency)
- **STEVE** (discrete visual representations)
- **Silicon-Menagerie** (SAYCam pretrained ViT)

All integrated directly into the SlotContrast repository.

## Installation

```bash
cd /work/hslee/egocentric-slotcontrast

# Install SlotContrast dependencies
poetry install

# Verify STEVE and silicon-menagerie repos exist
ls ../steve  # Should exist
ls ../silicon-menagerie  # Should exist
```

## Training in 3 Steps

### Step 1: Update Data Paths

```bash
vim configs/integrated_v2_saycam.yml

# Change these lines:
# dataset:
#   json_dir: "/path/to/saycam/json"  # ← YOUR PATH
#   img_dir: "/path/to/saycam/frames"  # ← YOUR PATH
```

### Step 2: Start Training

```bash
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/saycam \
    --log-dir ./logs
```

### Step 3: Monitor

```bash
# In another terminal
tensorboard --logdir ./logs

# Open browser: http://localhost:6006
```

## What Gets Trained

| Component | Status |
|-----------|--------|
| Silicon-Menagerie ViT | ❄️ **FROZEN** |
| Feature transform MLP | 🔥 **TRAINABLE** |
| Slot attention | 🔥 **TRAINABLE** |
| Temporal predictor | 🔥 **TRAINABLE** |
| Feature decoder | 🔥 **TRAINABLE** |
| STEVE dVAE | 🔥 **TRAINABLE** |
| STEVE decoder | 🔥 **TRAINABLE** |

## Expected Training Time

- **Single V100 GPU**: 24-36 hours
- **Single A100 GPU**: 18-24 hours
- **4x V100 GPUs**: 8-12 hours

## Configuration Options

### Use Different Components

```bash
# Only SlotContrast + Silicon ViT (faster)
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.USE_DVAE=false \
    globals.USE_STEVE_DECODER=false

# Only SlotContrast + STEVE (no pretrained ViT)
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.USE_SILICON_VIT=false

# Full integration (default)
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data
```

### Adjust for Your GPU

```bash
# If you get OOM errors
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.BATCH_SIZE_PER_GPU=8 \
    dataset.ep_length=6

# If you have more GPUs
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.NUM_GPUS=4
```

### Change Number of Slots

```bash
# More slots for complex scenes
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.NUM_SLOTS=10

# Fewer slots for simple scenes
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.NUM_SLOTS=5
```

## Monitor Training

Watch these metrics in TensorBoard:

| Metric | Good Range |
|--------|-----------|
| `train/loss_featrec` | 0.15-0.25 |
| `train/loss_ss` | 0.3-0.5 |
| `train/loss_dvae_recon` | 0.05-0.15 |
| `train/loss_steve_ce` | 2.0-4.0 |

If losses are much higher, check your data paths!

## Resume Training

```bash
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    --continue ./logs/experiment_name/checkpoints/step=50000.ckpt
```

## After Training

### Load Model

```python
from slotcontrast.models_steve import ObjectCentricModelWithSTEVE

model = ObjectCentricModelWithSTEVE.load_from_checkpoint(
    'logs/experiment/checkpoints/step=100000.ckpt'
)
model.eval()
```

### Extract Features

```python
import torch

# Load video
video = torch.randn(1, 8, 3, 224, 224)  # (B, T, C, H, W)

# Extract features
with torch.no_grad():
    outputs = model({"video": video})

# Get slots
slots = outputs['processor']['state']  # (1, 8, 7, 128)

# Get discrete tokens
tokens = outputs['dvae']['z_hard']  # (1, 8, H, W, 4096)

# Get reconstructions
dvae_recon = outputs['dvae']['reconstruction']  # (1, 8, 3, 224, 224)
```

## File Locations

All new code is in:
```
slotcontrast/modules/steve_components/  # STEVE components
slotcontrast/modules/silicon_vit.py     # Silicon ViT wrapper
slotcontrast/models_steve.py            # Extended model
configs/integrated_v2_saycam.yml        # Config
train_integrated_v2.py                  # Training script
```

Original repos (unchanged):
```
../steve/                 # STEVE (reference only)
../silicon-menagerie/     # Pretrained models
```

## Common Issues

### "Cannot import silicon-menagerie"
```bash
# Check path
ls ../silicon-menagerie

# Or disable silicon ViT
# Edit config: USE_SILICON_VIT: false
```

### "Out of memory"
```bash
# Reduce batch size
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    globals.BATCH_SIZE_PER_GPU=8
```

### "Loss is NaN"
```bash
# Reduce learning rate
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    optimizer.lr=0.0002
```

## Documentation

- **`INTEGRATED_V2_README.md`** - Complete guide
- **`INTEGRATED_V2_SUMMARY.md`** - Implementation details
- **`QUICK_START_V2.md`** - This file

## Example: Full Training Command

```bash
# Complete example with common options
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /data/saycam \
    --log-dir ./logs \
    --use-optimizations \
    globals.NUM_SLOTS=7 \
    globals.BATCH_SIZE_PER_GPU=16 \
    globals.NUM_GPUS=1 \
    trainer.max_steps=100000
```

That's it! Your integrated model should now be training. 🚀
