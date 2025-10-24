# Integrated Model V2: SlotContrast + STEVE + Silicon-Menagerie

## Overview

The Integrated Model V2 combines three powerful frameworks into a unified training pipeline:

1. **SlotContrast** - Temporal consistency via slot-based contrastive learning
2. **STEVE** - Discrete visual representations via dVAE and transformer decoder
3. **Silicon-Menagerie** - Pretrained ViT backbones trained on SAYCam egocentric videos

This integration happens **directly in the SlotContrast repository** without modifying the original STEVE or silicon-menagerie code.

## Architecture

```
Egocentric Video (B, T, 3, 224, 224)
         │
         ├─────────────────────────────────┐
         │                                 │
         v                                 v
┌─────────────────────┐          ┌──────────────────┐
│ Silicon-Menagerie   │          │  STEVE dVAE      │
│ Pretrained ViT      │          │  Encoder         │
│ (Frozen)            │          │  (Trainable)     │
└─────────────────────┘          └──────────────────┘
         │                                 │
         v                                 v
  ViT Features                    Discrete Tokens
  (B, T, 256, 768)                (B, T, H, W, 4096)
         │                                 │
         v                                 │
┌─────────────────────┐                   │
│ Feature Transform   │                   │
│ MLP: 768 → 128      │                   │
└─────────────────────┘                   │
         │                                 │
         v                                 │
┌─────────────────────┐                   │
│ Slot Attention      │                   │
│ (SlotContrast)      │                   │
│ → 7 Object Slots    │                   │
└─────────────────────┘                   │
         │                                 │
         v                                 │
  Slots (B, T, 7, 128)                    │
         │                                 │
         ├────────────┬────────────────────┤
         v            v                    v
┌─────────────┐  ┌──────────┐    ┌────────────────┐
│ Feature     │  │ Temporal │    │ STEVE Decoder  │
│ Decoder     │  │ Predictor│    │ (Transformer)  │
│ (MLP)       │  │ (Trans.) │    │                │
└─────────────┘  └──────────┘    └────────────────┘
         │            │                    │
         v            v                    v
┌─────────────┐  ┌──────────┐    ┌────────────────┐
│ Loss:       │  │ Loss:    │    │ Loss:          │
│ Feature     │  │ Slot-    │    │ Cross-Entropy  │
│ Recon (MSE) │  │ Contrast │    │                │
└─────────────┘  └──────────┘    └────────────────┘
                                          │
                                          v
                                  ┌────────────────┐
                                  │ dVAE Decoder   │
                                  │                │
                                  └────────────────┘
                                          │
                                          v
                                  ┌────────────────┐
                                  │ Loss: dVAE     │
                                  │ Recon (MSE)    │
                                  └────────────────┘
```

## What Was Integrated

### Files Added to SlotContrast Repository

```
slotcontrast/
├── modules/
│   ├── steve_components/
│   │   ├── __init__.py              # STEVE components package
│   │   ├── dvae.py                  # dVAE and STEVE decoder
│   │   └── steve_losses.py          # dVAE and cross-entropy losses
│   └── silicon_vit.py               # Silicon-menagerie ViT backbone
├── models_steve.py                  # Extended model with STEVE components
└── ...

configs/
└── integrated_v2_saycam.yml         # Training configuration

train_integrated_v2.py               # Training script
INTEGRATED_V2_README.md              # This file
```

### Key Components

**1. dVAE Module** (`slotcontrast/modules/steve_components/dvae.py`):
- Discrete VAE for learning discrete visual representations
- Encoder: Image → Discrete tokens (Gumbel-Softmax)
- Decoder: Discrete tokens → Image reconstruction
- Compatible with video inputs via automatic batching

**2. STEVE Decoder** (`slotcontrast/modules/steve_components/dvae.py`):
- Transformer decoder: Slots → Discrete tokens
- Autoregressive token prediction
- Cross-entropy loss for training

**3. STEVE Losses** (`slotcontrast/modules/steve_components/steve_losses.py`):
- `DVAEReconstructionLoss`: MSE between original and dVAE-reconstructed images
- `STEVECrossEntropyLoss`: Cross-entropy for discrete token prediction
- `STEVEGumbelTemperatureScheduler`: Anneals Gumbel temperature during training

**4. Silicon-Menagerie ViT** (`slotcontrast/modules/silicon_vit.py`):
- Wrapper for silicon-menagerie pretrained models
- Compatible with SlotContrast's encoder framework
- Supports all available SAYCam models

**5. Extended Model** (`slotcontrast/models_steve.py`):
- `ObjectCentricModelWithSTEVE`: Extends SlotContrast's ObjectCentricModel
- Integrates all STEVE components
- Manages multiple loss objectives
- Compatible with PyTorch Lightning

## Training

### Quick Start

```bash
cd /work/hslee/egocentric-slotcontrast

# 1. Update data paths in config
vim configs/integrated_v2_saycam.yml
# Set: dataset.json_dir and dataset.img_dir

# 2. Start training
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/saycam \
    --log-dir ./logs

# 3. Monitor training
tensorboard --logdir ./logs
```

### Configuration Options

**Enable/Disable Components:**

```yaml
globals:
  USE_SILICON_VIT: true  # Use silicon-menagerie (false for DINOv2)
  SILICON_MODEL: dino_say_vitb14  # Model name
  USE_DVAE: true  # Enable STEVE's dVAE
  USE_STEVE_DECODER: true  # Enable STEVE's transformer decoder
```

**Loss Weights:**

```yaml
model:
  loss_weights:
    loss_featrec: 1.0  # SlotContrast feature reconstruction
    loss_ss: 0.5  # SlotContrast temporal consistency
    loss_dvae_recon: 0.5  # STEVE dVAE reconstruction
    loss_steve_ce: 1.0  # STEVE cross-entropy
```

**Slot Configuration:**

```yaml
globals:
  NUM_SLOTS: 7  # Number of object slots
  SLOT_DIM: 128  # Slot dimension
  VOCAB_SIZE: 4096  # dVAE vocabulary size
```

### Training Strategies

#### Strategy 1: Full Integration (Default)

All components enabled with balanced loss weights.

```yaml
USE_SILICON_VIT: true
USE_DVAE: true
USE_STEVE_DECODER: true

loss_weights:
  loss_featrec: 1.0
  loss_ss: 0.5
  loss_dvae_recon: 0.5
  loss_steve_ce: 1.0
```

**Best for:**
- Maximum representation learning
- Rich object-centric features
- Downstream tasks requiring both continuous and discrete representations

**Training time:** ~24-36 hours on single V100

#### Strategy 2: SlotContrast + Silicon ViT (No STEVE)

Focus on temporal consistency with pretrained features.

```yaml
USE_SILICON_VIT: true
USE_DVAE: false
USE_STEVE_DECODER: false

loss_weights:
  loss_featrec: 1.0
  loss_ss: 0.5
```

**Best for:**
- Faster training
- When discrete representations not needed
- Pure temporal consistency learning

**Training time:** ~18-24 hours on single V100

#### Strategy 3: SlotContrast + STEVE (No Pretrained ViT)

Learn everything from scratch.

```yaml
USE_SILICON_VIT: false  # Use DINOv2 or ImageNet pretrained
USE_DVAE: true
USE_STEVE_DECODER: true

loss_weights:
  loss_featrec: 1.0
  loss_ss: 0.5
  loss_dvae_recon: 0.5
  loss_steve_ce: 1.0
```

**Best for:**
- Non-egocentric videos
- When SAYCam pretraining not relevant
- ImageNet-style datasets

**Training time:** ~24-30 hours on single V100

### Command Line Options

```bash
# Basic training
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs

# Override config parameters
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.NUM_SLOTS=10 \
    globals.USE_DVAE=false \
    model.loss_weights.loss_ss=1.0

# Resume from checkpoint
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    --continue ./logs/experiment/checkpoints/step=50000.ckpt

# Multi-GPU training
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    globals.NUM_GPUS=4 \
    globals.BATCH_SIZE_PER_GPU=8

# Enable optimizations
python train_integrated_v2.py configs/integrated_v2_saycam.yml \
    --data-dir /path/to/data \
    --use-optimizations
```

## Monitoring Training

### TensorBoard Metrics

```bash
tensorboard --logdir ./logs
```

**Key metrics to watch:**

1. **SlotContrast Losses:**
   - `train/loss_featrec` - Feature reconstruction (target: ~0.15-0.25)
   - `train/loss_ss` - Temporal consistency (target: ~0.3-0.5)

2. **STEVE Losses:**
   - `train/loss_dvae_recon` - dVAE reconstruction (target: ~0.05-0.15)
   - `train/loss_steve_ce` - Cross-entropy (target: ~2.0-4.0)

3. **Overall:**
   - `train/loss` - Total weighted loss (should decrease)

### Visualizations

Saved every 5000 steps in TensorBoard:
- Original video frames
- Slot attention maps
- dVAE reconstructions
- Discrete token visualizations

## Expected Results

After 100K steps with default configuration:

| Metric | Expected Value |
|--------|---------------|
| Feature Reconstruction | 0.15-0.25 |
| Temporal Consistency | 0.3-0.5 |
| dVAE Reconstruction | 0.05-0.15 |
| Cross-Entropy | 2.0-4.0 |
| Training Time (V100) | 24-36 hours |

**Qualitative results:**
- Slots consistently track same objects across frames
- dVAE produces clean reconstructions
- Discrete tokens capture visual patterns
- Temporal dynamics are smooth

## Using Trained Model

### Load Checkpoint

```python
from slotcontrast.models_steve import ObjectCentricModelWithSTEVE

# Load model
model = ObjectCentricModelWithSTEVE.load_from_checkpoint(
    'logs/experiment/checkpoints/step=100000.ckpt'
)
model.eval()

# Extract features
with torch.no_grad():
    outputs = model({"video": video_tensor})

# Access components
slots = outputs['processor']['state']  # (B, T, num_slots, slot_dim)
dvae_recon = outputs['dvae']['reconstruction']  # (B, T, C, H, W)
discrete_tokens = outputs['dvae']['z_hard']  # (B, T, H, W, vocab_size)
```

### Extract Different Representations

```python
# Continuous slot features
slots = outputs['processor']['state']  # For downstream tasks

# Discrete visual tokens
tokens = outputs['dvae']['z_hard']  # For generative modeling

# ViT features
vit_features = outputs['encoder']['backbone_features']  # For transfer learning

# Attention maps
attention = outputs['processor']['corrector']['masks']  # For visualization
```

## Comparison with Other Approaches

### vs. Standard SlotContrast

**Integrated V2 adds:**
- ✅ Discrete visual representations (dVAE)
- ✅ Autoregressive token modeling
- ✅ Optional SAYCam pretrained backbone
- ✅ Additional reconstruction supervision

**Trade-offs:**
- ⏱️ Slower training (~1.5x)
- 💾 More memory usage
- 🎯 Richer representations

### vs. STEVE

**Integrated V2 adds:**
- ✅ Temporal consistency via contrastive loss
- ✅ Better object tracking across frames
- ✅ Pretrained visual features
- ✅ SlotContrast's training infrastructure

**Trade-offs:**
- 🔧 More complex configuration
- ⏱️ Longer training time
- 📊 More metrics to monitor

### vs. Integrated V1

**V2 improvements:**
- ✅ Native SlotContrast integration (no wrapper)
- ✅ STEVE losses properly integrated
- ✅ Easier configuration
- ✅ Better compatibility

**V1 advantages:**
- 🚀 Simpler standalone model
- 📚 More examples and documentation

## Troubleshooting

### Issue: Import errors for STEVE components

**Solution:**
```bash
# Ensure files are in correct location
ls slotcontrast/modules/steve_components/
# Should show: __init__.py, dvae.py, steve_losses.py
```

### Issue: Silicon-menagerie models not loading

**Solution:**
```bash
# Check silicon-menagerie repository exists
ls ../silicon-menagerie

# Or disable silicon ViT
# In config: USE_SILICON_VIT: false
```

### Issue: High memory usage

**Solution:**
```yaml
# Reduce batch size
globals.BATCH_SIZE_PER_GPU: 8

# Reduce video length
dataset.ep_length: 6

# Disable dVAE temporarily
globals.USE_DVAE: false
```

### Issue: Loss is NaN

**Solution:**
```yaml
# Reduce learning rate
optimizer.lr: 0.0002

# Increase gradient clipping
trainer.gradient_clip_val: 0.1

# Reduce cross-entropy weight
model.loss_weights.loss_steve_ce: 0.5
```

### Issue: Slots collapse

**Solution:**
```yaml
# Increase slot attention iterations
grouper.n_iters: 5

# Reduce contrastive temperature
losses.loss_ss.temperature: 0.05

# Increase contrastive weight
model.loss_weights.loss_ss: 1.0
```

## Advanced Usage

### Custom Loss Weights

Experiment with different loss balances:

```yaml
# Emphasize temporal consistency
model.loss_weights:
  loss_featrec: 0.5
  loss_ss: 2.0  # Higher weight
  loss_dvae_recon: 0.3
  loss_steve_ce: 0.5

# Emphasize discrete representations
model.loss_weights:
  loss_featrec: 0.5
  loss_ss: 0.3
  loss_dvae_recon: 1.0  # Higher weight
  loss_steve_ce: 2.0  # Higher weight
```

### Custom Vocabulary Size

```yaml
globals:
  VOCAB_SIZE: 8192  # Larger vocabulary for more details
  # or
  VOCAB_SIZE: 2048  # Smaller for faster training
```

### Different Pretrained Models

```yaml
globals:
  # SAYCam models
  SILICON_MODEL: dino_say_vitb14
  # SILICON_MODEL: dino_s_vitb14  # Baby S only
  # SILICON_MODEL: dino_a_vitb14  # Baby A only
  # SILICON_MODEL: mae_say_vitb14  # MAE pretrained
```

## Citation

If you use this integrated model, please cite all three works:

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
    title={Learning high-level visual representations from a child's perspective},
    author={Orhan, AE and Lake, BM},
    journal={arXiv:2305.15372},
    year={2023}
}
```

## Summary

Integrated Model V2 provides a powerful combination of:
- ✅ Temporal consistency (SlotContrast)
- ✅ Discrete representations (STEVE)
- ✅ Pretrained features (Silicon-Menagerie)
- ✅ Easy configuration and training
- ✅ Native SlotContrast integration

All within the SlotContrast repository, ready to train on your egocentric video data!
