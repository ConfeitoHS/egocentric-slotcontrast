# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**SlotContrast** is a research codebase for temporally consistent object-centric learning from videos using slot-based contrastive learning (CVPR 2025 Oral). The repository includes:

1. Core SlotContrast framework for video object discovery
2. Integrated model combining STEVE + Silicon-Menagerie pretrained models
3. Support for synthetic (MOVi-C, MOVi-E) and real-world (YT-VIS, SAYCam) datasets

Key concept: The model decomposes video frames into "slots" where each slot represents an object or background element, with temporal contrastive loss ensuring slots consistently track the same objects across frames.

## Repository Structure

```
slotcontrast/          # Core framework (PyTorch Lightning)
├── models.py          # ObjectCentricModel (main training model)
├── train.py           # Training entry point
├── modules/           # Model components (encoders, groupers, decoders)
├── losses.py          # Contrastive and reconstruction losses
├── data/              # Data loading pipelines
└── metrics.py         # Evaluation metrics (ARI, mBO)

integrated_model/      # Combined STEVE + Silicon-Menagerie model
├── training/          # Training infrastructure
└── examples/          # Usage examples

configs/               # YAML configurations
├── slotcontrast/      # SlotContrast training configs
└── *.yml              # SAYCam dataset configs

data/                  # Dataset preparation scripts
tests/                 # Test suite
```

## Common Commands

### Setup
```bash
# Install dependencies (use Poetry)
poetry install

# Optional extras
poetry install -E tensorflow    # For MOVi dataset conversion
poetry install -E coco          # For COCO/YT-VIS datasets
poetry install -E notebook      # For Jupyter notebooks

# Run tests
poetry run pytest
poetry run pytest -m "not slow"  # Skip slow tests
```

### Training

**SlotContrast (original framework):**
```bash
# Basic training
poetry run python -m slotcontrast.train configs/slotcontrast/movi_c.yml

# With custom data/log directories
poetry run python -m slotcontrast.train configs/slotcontrast/movi_c.yml \
    --data-dir /path/to/data \
    --log-dir ./logs

# Resume training
poetry run python -m slotcontrast.train \
    --continue <path_to_checkpoint_or_log_dir> \
    configs/slotcontrast/movi_c.yml

# Enable optimizations (TF32, cudnn benchmark)
poetry run python -m slotcontrast.train \
    --use-optimizations \
    configs/slotcontrast/movi_c.yml

# Override config parameters
poetry run python -m slotcontrast.train configs/slotcontrast/movi_c.yml \
    globals.NUM_SLOTS=10 \
    trainer.max_steps=50000 \
    optimizer.lr=0.0002
```

**Integrated Model (STEVE + Silicon-Menagerie):**
```bash
cd integrated_model/training

python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/saycam/data \
    --log-dir ./logs
```

### Testing
```bash
# Run quick test to verify setup
poetry run python -m slotcontrast.train tests/configs/test_dummy_image.yml

# Run specific test
poetry run pytest tests/test_train.py -v

# Run with dataset dependencies
poetry run pytest -m dataset
```

### Inference
```bash
# Run inference on videos with pretrained checkpoint
poetry run python -m slotcontrast.inference --config configs/inference/movi_c.yml

# Update checkpoint path in config first:
# checkpoint: path/to/slotcontrast-movi-c.ckpt
```

### Dataset Preparation

**MOVi datasets:**
```bash
poetry install -E tensorflow
python data/save_movi.py --level c --split train --maxcount 32 --only-video data/movi_c
python data/save_movi.py --level e --split validation --maxcount 32 data/movi_e
```

**COCO:**
```bash
poetry install -E coco
python data/save_coco.py --split train --maxcount 128 --only-images --out-path data/coco
```

**YT-VIS:**
```bash
poetry install -E coco
python data/save_ytvis2021.py --split train --maxcount 32 --only-videos --resize --out-path data/ytvis2021_resized
```

## Architecture

### Core Model Flow (slotcontrast/models.py)

```
Input Video (B, T, C, H, W)
    ↓
Encoder (ViT backbone, e.g., DINOv2)
    ↓
Features (B, T, N_patches, feat_dim)
    ↓
Initializer → Initial Slots (B, num_slots, slot_dim)
    ↓
Processor (SlotAttention + Temporal Predictor)
    ├─ Corrector: Slot attention over features
    ├─ Predictor: Transformer for temporal modeling
    └─ State: Refined slots (B, T, num_slots, slot_dim)
    ↓
Decoder (MLP) → Reconstructed features
    ↓
Losses:
├─ Feature Reconstruction (MSE)
└─ Slot-Slot Contrastive (temporal consistency)
```

### Key Components

**ObjectCentricModel** (slotcontrast/models.py:147): Main PyTorch Lightning module
- Manages encoder → grouper → decoder pipeline
- Computes reconstruction + contrastive losses
- Handles video/image inputs with MapOverTime/ScanOverTime wrappers

**modules/** directory:
- `encoders.py`: Vision backbone wrappers (TIMM models, DINOv2)
- `groupers.py`: SlotAttention implementation
- `video.py`: LatentProcessor for temporal modeling
- `decoders.py`: Reconstruction decoders
- `networks.py`: Transformer, MLP building blocks
- `timm.py`: TIMM model integration with feature extraction

**Video Processing:**
- `MapOverTime`: Apply module independently to each frame
- `ScanOverTime`: Process frames sequentially (for recurrent operations)
- `LatentProcessor`: Combines corrector (SlotAttention) + predictor (Transformer)

### Configuration System

Configs use OmegaConf with variable interpolation:

```yaml
globals:
  NUM_SLOTS: 7
  SLOT_DIM: 128
  FEAT_DIM: 768
  NUM_GPUS: 1
  BATCH_SIZE_PER_GPU: 32
  TOTAL_BATCH_SIZE: "${mul: ${.NUM_GPUS}, ${.BATCH_SIZE_PER_GPU}}"

model:
  input_type: video  # or "image"
  encoder:
    backbone:
      name: TimmExtractor
      model: vit_base_patch14_dinov2.lvd142m  # or dino_say_vitb14 for SAYCam
      frozen: true
    output_transform:
      name: networks.two_layer_mlp
      inp_dim: ${globals.FEAT_DIM}
      outp_dim: ${globals.SLOT_DIM}

  grouper:
    name: SlotAttention
    inp_dim: ${globals.SLOT_DIM}
    slot_dim: ${globals.SLOT_DIM}
    n_iters: 3  # Number of attention iterations

  decoder:
    name: MLPDecoder
    inp_dim: ${globals.SLOT_DIM}
    outp_dim: ${globals.FEAT_DIM}
    hidden_dims: [1024, 1024, 1024]

  losses:
    loss_featrec:
      name: MSELoss
    loss_ss:  # Slot-slot contrastive
      name: Slot_Slot_Contrastive_Loss
      temperature: 0.1

  loss_weights:
    loss_featrec: 1.0
    loss_ss: 0.5  # Key hyperparameter for temporal consistency
```

### Loss Functions

**Slot-Slot Contrastive Loss** (loss_ss): The key innovation
- Encourages same-object slots across frames to be similar
- Discourages different-object slots from being similar
- Temperature parameter controls contrastiveness (lower = harder negatives)

**Feature Reconstruction Loss** (loss_featrec): Standard MSE
- Reconstructs ViT backbone features (not pixels)
- Provides pixel-level supervision through feature space

### Data Pipeline

**DataModules** (slotcontrast/data/datamodules.py):
- Return batches as dictionaries: `{"video": tensor, "segmentations": tensor, ...}`
- Video shape: (B, T, C, H, W)
- Segmentations for validation metrics (if available)

**Key datasets:**
- `MOViDataModule`: MOVi-C/E synthetic datasets
- `YTVISDataModule`: YouTube-VIS real-world videos
- `SAYCAMDataModule`: Egocentric child-view videos

**Data location**: Set via environment variable `SLOTCONTRAST_DATA_PATH` or `--data-dir` flag

## Integrated Model (STEVE + Silicon-Menagerie)

Located in `integrated_model/` directory. Combines:
1. **Silicon-Menagerie**: Pretrained ViT on SAYCam (child egocentric videos)
2. **STEVE**: Slot attention mechanism for object decomposition
3. **SlotContrast**: Temporal consistency via contrastive learning

**Key difference from core SlotContrast:**
- Uses pretrained SAYCam ViT instead of DINOv2/ImageNet models
- Focuses on egocentric video understanding
- Provides downstream task implementations (4-way classification, linear probing, few-shot)

**Training:**
```bash
cd integrated_model/training
python train_integrated.py integrated_saycam.yml \
    --data-dir /path/to/saycam \
    --log-dir ./logs
```

See `integrated_model/README.md` and `TRAINING_QUICK_START.md` for details.

## Development Workflow

### Adding New Models

1. Implement module in `slotcontrast/modules/` (encoders, groupers, decoders)
2. Register in corresponding `build_*()` function
3. Add config in `configs/`
4. Test with `tests/configs/test_*.yml`

### Adding New Losses

1. Implement in `slotcontrast/losses.py` inheriting from `Loss` class
2. Define `get_prediction()` and `get_target()` methods
3. Register in `build()` function
4. Add to config under `model.losses`

### Adding New Metrics

1. Implement in `slotcontrast/metrics.py` inheriting from `torchmetrics.Metric`
2. Register in `build()` function
3. Add to config under `val_metrics` or `train_metrics`

### Code Style

```bash
# Format code
poetry run black slotcontrast tests
poetry run isort slotcontrast tests

# Lint
poetry run ruff slotcontrast tests

# Pre-commit hooks (recommended)
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Important Implementation Details

### Video Input Handling
- Video models wrap frame-level modules with `MapOverTime` or `ScanOverTime`
- `input_type: video` triggers automatic wrapping in model builder
- Temporal processing happens in `LatentProcessor` via `ScanOverTime`

### Slot Attention Iterations
- `grouper.n_iters` controls refinement iterations (typically 2-5)
- More iterations = better object separation but slower
- First iteration (corrector) can have different `n_iters` via `latent_processor.first_step_corrector_args`

### Checkpoint Management
- Checkpoints saved every N steps (config: `checkpoint_every_n_steps`)
- Auto-resume from last checkpoint in log directory
- Use `--continue <path>` to explicitly resume from checkpoint or log directory

### Multi-GPU Training
- Automatically uses DDP strategy if multiple GPUs available
- Set `trainer.devices` in config or let system auto-detect
- Effective batch size = `NUM_GPUS * BATCH_SIZE_PER_GPU`

### Memory Optimization
If OOM errors occur:
1. Reduce `BATCH_SIZE_PER_GPU`
2. Reduce video length: `dataset.ep_length`
3. Reduce number of slots: `globals.NUM_SLOTS`
4. Freeze backbone: `encoder.backbone.frozen: true`
5. Enable mixed precision: `trainer.precision: 16`

### Validation Metrics
- Video ARI: Adjusted Rand Index (object segmentation quality)
- Video mBO: Mean Best Overlap (tracking consistency)
- Requires ground truth segmentation masks
- Computed on validation set every `val_check_interval` steps

## Pretrained Checkpoints

Available on HuggingFace:
- MOVi-C: https://huggingface.co/annamanasyan/slotcontrast/resolve/main/movi_c.ckpt
- MOVi-E: https://huggingface.co/annamanasyan/slotcontrast/resolve/main/movi_e.ckpt
- YT-VIS: https://huggingface.co/annamanasyan/slotcontrast/resolve/main/ytvis.ckpt

Load for inference:
```python
from slotcontrast.models import ObjectCentricModel
model = ObjectCentricModel.load_from_checkpoint("path/to/checkpoint.ckpt")
```

## Common Issues

**ImportError with TIMM models**: Update timm: `pip install --upgrade timm`

**DataLoader workers crash**: Reduce `dataset.num_workers` or set to 0

**Loss is NaN**:
- Reduce learning rate
- Increase gradient clipping: `trainer.gradient_clip_val`
- Check input normalization

**Slots collapse**: All slots learn same representation
- Increase slot attention iterations: `grouper.n_iters`
- Reduce contrastive temperature: `losses.loss_ss.temperature`
- Check initialization: try `FixedLearnedInit` vs `GaussianInit`

**Slow training**:
- Enable optimizations: `--use-optimizations`
- Increase `dataset.num_workers`
- Freeze backbone: `encoder.backbone.frozen: true`

## Related Projects

This codebase is adapted from **Videosaur**: https://github.com/martius-lab/videosaur

Required external repositories for integrated model:
- STEVE: /work/hslee/steve
- Silicon-Menagerie: /work/hslee/silicon-menagerie

## Citation

```bibtex
@inproceedings{manasyan2025temporally,
    title={Temporally Consistent Object-Centric Learning by Contrasting Slots},
    author={Manasyan, Anna and Seitzer, Maximilian and Radovic, Filip and Martius, Georg and Zadaianchuk, Andrii},
    booktitle={CVPR},
    year={2025}
}
```
