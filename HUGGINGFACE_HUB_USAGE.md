# Using Hugging Face Hub Models with SlotContrast

This guide shows how to load pretrained models from Hugging Face Hub as encoder backbones.

## Method 1: Using TimmExtractor (Recommended)

### Load Standard TIMM Models from HF Hub

```yaml
model:
  encoder:
    backbone:
      name: TimmExtractor
      model: "hf-hub:timm/vit_base_patch16_224.augreg_in21k"
      pretrained: true
      frozen: true
      features:
        - vit_block12
        - vit_block_keys12
```

### Load Facebook Models (DINOv2, DINO, MAE)

```yaml
# DINOv2 Base
model:
  encoder:
    backbone:
      name: TimmExtractor
      model: "hf-hub:facebook/dinov2-base"
      pretrained: true
      frozen: true
      features:
        - vit_block12
```

```yaml
# DINOv2 Large
model:
  encoder:
    backbone:
      name: TimmExtractor
      model: "hf-hub:facebook/dinov2-large"
      pretrained: true
      frozen: true
```

```yaml
# Original DINO ViT-B/16
model:
  encoder:
    backbone:
      name: TimmExtractor
      model: "hf-hub:facebook/dino-vitb16"
      pretrained: true
      frozen: true
```

## Method 2: Using SiliconViTExtractor with HF Hub

For Silicon-Menagerie or custom models on HF Hub:

```yaml
model:
  encoder:
    use_silicon_vit: true
    backbone:
      model: dino_say_vitb14  # Fallback name
      frozen: true
      hf_hub_id: "your-username/silicon-menagerie-saycam"  # HF Hub ID
```

### Example: Using Your Custom Model

If you uploaded your Silicon-Menagerie model to HF Hub:

```yaml
model:
  encoder:
    use_silicon_vit: true
    backbone:
      model: dino_say_vitb14
      frozen: true
      hf_hub_id: "hslee/silicon-menagerie-saycam-vitb14"  # Your HF model
```

## Method 3: Load Custom Checkpoint from HF Hub

```yaml
model:
  encoder:
    backbone:
      name: TimmExtractor
      model: "vit_base_patch16_224"
      pretrained: false
      checkpoint_path: "hf-hub:username/model-name/checkpoint.pth"
      frozen: true
```

## Complete Config Example

Here's a complete config using DINOv2 from HF Hub:

```yaml
experiment_group: slotcontrast_hf
experiment_name: movi_e_dinov2_hf

globals:
  NUM_SLOTS: 15
  SLOT_DIM: 128
  FEAT_DIM: 768  # DINOv2 ViT-B feature dimension
  NUM_PATCHES: 576  # 24x24 patches

trainer:
  max_steps: 300000
  precision: bf16-mixed

model:
  input_type: video

  encoder:
    use_silicon_vit: false  # Use TimmExtractor instead

    backbone:
      name: TimmExtractor
      model: "hf-hub:facebook/dinov2-base"  # Load from HF Hub
      pretrained: true
      frozen: true
      features:
        - vit_block12
        - vit_block_keys12
      model_kwargs:
        dynamic_img_size: true

    output_transform:
      name: networks.two_layer_mlp
      inp_dim: ${globals.FEAT_DIM}
      outp_dim: ${globals.SLOT_DIM}
      hidden_dim: 1536
      layer_norm: true

  # ... rest of config
```

## Available Models on HF Hub

### DINOv2 (Meta AI)
- `hf-hub:facebook/dinov2-small` (384 dim)
- `hf-hub:facebook/dinov2-base` (768 dim)
- `hf-hub:facebook/dinov2-large` (1024 dim)
- `hf-hub:facebook/dinov2-giant` (1536 dim)

### DINO (Original)
- `hf-hub:facebook/dino-vits16` (384 dim)
- `hf-hub:facebook/dino-vits8` (384 dim)
- `hf-hub:facebook/dino-vitb16` (768 dim)
- `hf-hub:facebook/dino-vitb8` (768 dim)

### MAE (Masked Autoencoder)
- `hf-hub:facebook/vit-mae-base` (768 dim)
- `hf-hub:facebook/vit-mae-large` (1024 dim)
- `hf-hub:facebook/vit-mae-huge` (1280 dim)

### TIMM Models
- `hf-hub:timm/vit_base_patch16_224.augreg_in21k`
- `hf-hub:timm/vit_large_patch16_224.augreg_in21k`
- Browse more: https://huggingface.co/timm

## Programmatic Usage

### Python Code

```python
from slotcontrast.modules.silicon_vit import SiliconViTBackbone

# Load from HF Hub
backbone = SiliconViTBackbone(
    model_name='dino_say_vitb14',  # Fallback name
    hf_hub_id='facebook/dinov2-base',  # Load from HF
    frozen=True,
)

# Or use TimmExtractor
from slotcontrast.modules.encoders import TimmExtractor

encoder = TimmExtractor(
    model='hf-hub:facebook/dinov2-base',
    pretrained=True,
    frozen=True,
    features=['vit_block12'],
)
```

## Upload Your Own Model to HF Hub

### 1. Install Hugging Face CLI
```bash
pip install huggingface-hub
huggingface-cli login
```

### 2. Upload Model

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload your checkpoint
api.upload_file(
    path_or_fileobj="path/to/your/model.pth",
    path_in_repo="checkpoint.pth",
    repo_id="your-username/your-model-name",
    repo_type="model",
)
```

### 3. Use in Config

```yaml
model:
  encoder:
    backbone:
      name: TimmExtractor
      model: "vit_base_patch16_224"
      checkpoint_path: "hf-hub:your-username/your-model-name/checkpoint.pth"
```

## Benefits

1. **No Local Storage**: Models download on-demand
2. **Version Control**: HF Hub tracks model versions
3. **Easy Sharing**: Share models with team/community
4. **Caching**: Downloaded models cached automatically
5. **Fallback**: If HF download fails, can still use local models

## Troubleshooting

### Authentication Error

If you get authentication errors:
```bash
huggingface-cli login
# Enter your HF token
```

### Download Timeout

Increase timeout in code:
```python
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes
```

### Model Not Found

Check if model exists:
```bash
huggingface-cli repo-files facebook/dinov2-base
```

### Cache Location

Models cached at:
- Linux: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

Clear cache:
```bash
rm -rf ~/.cache/huggingface/hub/
```

## See Also

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [TIMM Documentation](https://huggingface.co/docs/timm/index)
- [DINOv2 Model Card](https://huggingface.co/facebook/dinov2-base)
