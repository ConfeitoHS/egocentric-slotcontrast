# Training the Integrated Model

This directory contains the training infrastructure for the integrated model that combines:
1. **Silicon-Menagerie** pretrained ViT (frozen backbone for visual features)
2. **STEVE** slot attention mechanism (trainable for object decomposition)
3. **SlotContrast** temporal consistency (trainable for video understanding)

## Architecture Overview

```
Egocentric Video (B, T, C, H, W)
         |
         v
┌────────────────────────────────────────┐
│  Silicon-Menagerie Pretrained ViT      │
│  (Frozen - dino_say_vitb14)            │
│  → Extracts rich visual features       │
└────────────────────────────────────────┘
         |
         v
    ViT Features (B, T, N, 768)
         |
         v
┌────────────────────────────────────────┐
│  Feature Transform (MLP)               │
│  → 768 → 128 dimensions                │
└────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────┐
│  STEVE Slot Attention (Trainable)      │
│  → Decomposes features into slots      │
│  → Each slot = one object/background   │
└────────────────────────────────────────┘
         |
         v
    Object Slots (B, T, num_slots, 128)
         |
         v
┌────────────────────────────────────────┐
│  SlotContrast Temporal Module          │
│  → Enforces temporal consistency        │
│  → Contrastive loss across time        │
└────────────────────────────────────────┘
         |
         v
┌────────────────────────────────────────┐
│  Decoder (MLP)                         │
│  → Reconstructs ViT features           │
│  → 128 → 768 dimensions                │
└────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
cd /work/hslee/egocentric-slotcontrast

# Install dependencies
poetry install

# Or use pip
pip install -r integrated_model/requirements.txt
```

### 2. Prepare Data

Update the dataset paths in `integrated_saycam.yml`:

```yaml
dataset:
  json_dir: "/path/to/your/saycam/json"  # Metadata
  img_dir: "/path/to/your/saycam/frames"  # Video frames
  img_size: 224
  ep_length: 8  # Frames per clip
  batch_size: 16
```

### 3. Start Training

```bash
cd integrated_model/training

# Basic training
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs

# Training with custom settings
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    --use-optimizations \
    dataset.batch_size=32 \
    globals.NUM_SLOTS=10

# Resume from checkpoint
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/data \
    --log-dir ./logs \
    --continue ./logs/experiment_name/checkpoints/step=10000.ckpt
```

## Configuration Details

### Key Hyperparameters

```yaml
globals:
  NUM_SLOTS: 7  # Number of object slots (adjust for your data)
  SLOT_DIM: 128  # Slot representation size
  SAYCAM_MODEL: dino_say_vitb14  # Pretrained model
  FEAT_DIM: 768  # ViT feature dimension

  # Adjust based on GPU memory
  BATCH_SIZE_PER_GPU: 16  # Reduce if OOM
  NUM_GPUS: 1

trainer:
  max_steps: 100000  # Total training steps
  gradient_clip_val: 0.05  # Gradient clipping
```

### Loss Configuration

The model is trained with two losses:

1. **Feature Reconstruction Loss** (`loss_featrec`):
   - Reconstructs ViT features from slots
   - Weight: 1.0
   - Type: MSE loss

2. **Slot-Slot Contrastive Loss** (`loss_ss`):
   - Enforces temporal consistency of slots
   - Weight: 0.5
   - Temperature: 0.1

```yaml
loss_weights:
  loss_featrec: 1.0  # Feature reconstruction
  loss_ss: 0.5  # Temporal consistency
```

### Model Components

**Encoder (Frozen):**
```yaml
encoder:
  backbone:
    name: TimmExtractor
    model: dino_say_vitb14
    frozen: true  # Keep pretrained weights
```

**Slot Attention (Trainable):**
```yaml
grouper:
  name: SlotAttention
  inp_dim: 128
  slot_dim: 128
  n_iters: 3  # Attention iterations
```

**Decoder (Trainable):**
```yaml
decoder:
  name: MLPDecoder
  inp_dim: 128
  outp_dim: 768
  hidden_dims: [1024, 1024, 1024]
```

## Training Strategies

### Strategy 1: Default (Recommended)
**Freeze backbone, train slots + temporal consistency**

```yaml
encoder:
  backbone:
    frozen: true  # Freeze ViT
grouper:
  frozen: false  # Train slot attention
```

**Pros:**
- Fast training
- Leverages pretrained features
- Good for limited data

**Training time:** ~12-24 hours on single GPU

### Strategy 2: Fine-tune Everything
**Train all components**

```yaml
encoder:
  backbone:
    frozen: false  # Finetune ViT
  learning_rate: 1e-5  # Low LR for backbone
```

**Pros:**
- Maximum performance
- Adapts to specific dataset

**Cons:**
- Slower training
- Requires more data
- Risk of overfitting

**Training time:** ~24-48 hours on single GPU

### Strategy 3: Two-Stage Training

**Stage 1:** Train slots with frozen backbone
```bash
python train_integrated.py integrated_saycam.yml \
    --log-dir ./logs/stage1 \
    trainer.max_steps=50000
```

**Stage 2:** Fine-tune with lower learning rate
```bash
python train_integrated.py integrated_saycam.yml \
    --log-dir ./logs/stage2 \
    --continue ./logs/stage1/checkpoints/step=50000.ckpt \
    optimizer.lr=0.0001 \
    trainer.max_steps=25000
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs

# Then open http://localhost:6006
```

**Metrics to watch:**
- `train/loss`: Overall training loss (should decrease)
- `train/loss_featrec`: Feature reconstruction (should converge to ~0.1-0.3)
- `train/loss_ss`: Contrastive loss (should decrease gradually)
- `val/loss`: Validation loss (check for overfitting)

### Checkpoints

Checkpoints are saved every 2500 steps in:
```
logs/experiment_name/checkpoints/
├── step=2500.ckpt
├── step=5000.ckpt
├── step=7500.ckpt
└── ...
```

### Visualizations

Slot attention visualizations are saved every 5000 steps in TensorBoard:
- Original video frames
- Attention maps for each slot
- Reconstructed features

## Advanced Configuration

### Adjust Number of Slots

```yaml
globals:
  NUM_SLOTS: 10  # Increase for complex scenes
```

**Guidelines:**
- 7 slots: Good for simple scenes (2-5 objects)
- 10-15 slots: Complex scenes (5-10 objects)
- More slots = more memory usage

### Custom Learning Rates

```yaml
optimizer:
  lr: 0.0004  # Base learning rate for slots

  # For different components
  param_groups:
    - regex: "encoder.*"
      lr: 0.00001  # Low LR for encoder
    - regex: "grouper.*"
      lr: 0.0004  # Higher LR for slots
    - regex: "decoder.*"
      lr: 0.0002  # Medium LR for decoder
```

### Multi-GPU Training

```yaml
globals:
  NUM_GPUS: 4
  BATCH_SIZE_PER_GPU: 16
  # Effective batch size = 4 * 16 = 64

trainer:
  devices: 4
  strategy: ddp  # Distributed data parallel
```

### Memory Optimization

If you encounter OOM errors:

```yaml
# Reduce batch size
globals:
  BATCH_SIZE_PER_GPU: 8

# Reduce video length
dataset:
  ep_length: 6  # Fewer frames

# Reduce slots
globals:
  NUM_SLOTS: 5

# Use gradient checkpointing (add to train_integrated.py)
```

## Evaluation

### During Training

Validation metrics are computed every 2500 steps. If you have segmentation masks:

```yaml
val_metrics:
  ari:
    name: VideoARI
    ignore_background: true
  mbo:
    name: VideoIoU
    matching: overlap
    ignore_background: true
```

### After Training

```bash
# Extract features from trained model
python extract_features.py \
    --checkpoint ./logs/experiment/checkpoints/step=100000.ckpt \
    --data-dir /path/to/test/data \
    --output-dir ./features

# Evaluate on downstream tasks
python evaluate_downstream.py \
    --features ./features \
    --task 4way_classification
```

## Troubleshooting

### Issue: Loss is NaN

**Solutions:**
1. Reduce learning rate: `optimizer.lr=0.0001`
2. Increase gradient clipping: `trainer.gradient_clip_val=0.1`
3. Check data for invalid values

### Issue: No improvement after many steps

**Solutions:**
1. Check if backbone is frozen: `encoder.backbone.frozen=true`
2. Increase contrastive loss weight: `model.loss_weights.loss_ss=1.0`
3. Try different initialization: `model.initializer.name=RandomInit`

### Issue: Slots collapse to same representation

**Solutions:**
1. Increase number of slot attention iterations: `grouper.n_iters=5`
2. Reduce contrastive temperature: `losses.loss_ss.temperature=0.05`
3. Add diversity regularization

### Issue: Out of memory

**Solutions:**
1. Reduce batch size: `dataset.batch_size=8`
2. Reduce video length: `dataset.ep_length=6`
3. Reduce number of slots: `globals.NUM_SLOTS=5`
4. Use mixed precision training: Add `trainer.precision=16` to config

## File Structure

```
training/
├── README.md                    # This file
├── train_integrated.py          # Main training script
├── integrated_saycam.yml        # Training configuration
├── custom_encoders.py           # Silicon-Menagerie encoder wrapper
├── steve_modules.py             # STEVE slot attention modules
└── utils/
    ├── visualization.py         # Visualization utilities
    └── evaluation.py            # Evaluation scripts
```

## Expected Results

After training for 100K steps with default settings:

- **Feature reconstruction loss:** ~0.15-0.25
- **Contrastive loss:** ~0.3-0.5
- **Slot consistency:** Slots should track same objects across frames
- **Training time:** ~18-24 hours on single V100 GPU

## Next Steps

1. **Evaluate on downstream tasks:**
   - 4-way object classification
   - Linear probing
   - Few-shot learning

2. **Visualize learned representations:**
   ```bash
   python visualize_slots.py --checkpoint ./logs/.../step=100000.ckpt
   ```

3. **Fine-tune for specific tasks:**
   ```bash
   python finetune_downstream.py \
       --checkpoint ./logs/.../step=100000.ckpt \
       --task object_recognition
   ```

## Citation

If you use this training code, please cite the original papers:

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

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the SlotContrast documentation
3. Open an issue in the repository
