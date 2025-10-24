# BF16 (Bfloat16) Mixed Precision Training

## What is BF16?

Bfloat16 (Brain Floating Point) is a 16-bit floating point format that:
- Uses the same exponent range as FP32 (8 bits) but reduced mantissa (7 bits)
- Provides better numerical stability than FP16
- Reduces memory usage by ~50%
- Speeds up training on modern GPUs (A100, H100, etc.)
- Requires less hyperparameter tuning than FP16

## Benefits

1. **Memory Savings**: ~50% less memory for activations and gradients
2. **Speed**: 2-3x faster on Ampere/Hopper GPUs (A100, H100)
3. **Stability**: Better than FP16, similar convergence to FP32
4. **No Loss Scaling**: Unlike FP16, doesn't require loss scaling

## How to Enable

### Option 1: Command Line Flag

```bash
python -m slotcontrast.train configs/integrated_v2_movi_e.yaml --bf16
```

### Option 2: Config File

Add to your YAML config:

```yaml
trainer:
  precision: bf16-mixed
```

### Option 3: Multi-GPU with torchrun

```bash
torchrun --nproc_per_node=8 -m slotcontrast.train \
    configs/integrated_v2_movi_e_lowmem.yaml \
    --bf16
```

## Example Configs with BF16

The following configs have BF16 enabled by default:
- `configs/integrated_v2_movi_e_lowmem.yaml` - Memory-optimized STEVE config

## Requirements

**Hardware:**
- NVIDIA Ampere GPUs (A100, A6000, RTX 3090, RTX 4090)
- NVIDIA Hopper GPUs (H100)
- AMD MI250X

**Software:**
- PyTorch >= 1.10
- CUDA >= 11.0

## Performance Comparison

| Precision | Memory Usage | Speed | Accuracy |
|-----------|--------------|-------|----------|
| FP32      | 100%         | 1x    | Baseline |
| BF16      | ~50%         | 2-3x  | ~Same    |
| FP16      | ~50%         | 2-3x  | Variable |

## GPU Compatibility

✅ **Supported (Native BF16):**
- NVIDIA A100
- NVIDIA H100
- NVIDIA RTX 3090/4090
- NVIDIA A6000
- AMD MI250X

⚠️ **Partial Support (Emulated, slower):**
- NVIDIA V100 (works but no speedup)
- NVIDIA RTX 2080 Ti (works but no speedup)

❌ **Not Supported:**
- Older GPUs (Pascal, Maxwell)

## Troubleshooting

### NaN/Inf Losses

If you encounter NaN losses with BF16:

```yaml
# Try gradient clipping
trainer:
  precision: bf16-mixed
  gradient_clip_val: 1.0  # Increase if needed
```

### No Speedup

Check GPU support:
```python
import torch
print(torch.cuda.is_bf16_supported())  # Should be True
```

### Accuracy Issues

BF16 should give similar results to FP32. If not:
1. Check if your GPU natively supports BF16
2. Try increasing gradient clipping
3. Verify batch size is adequate (BF16 allows larger batches)

## Best Practices

1. **Use with STEVE**: BF16 helps significantly with STEVE's memory-intensive dVAE
2. **Combine with other optimizations**:
   ```bash
   python -m slotcontrast.train config.yaml \
       --bf16 \
       --use-optimizations
   ```
3. **Increase batch size**: Use saved memory for larger batches
4. **Monitor metrics**: First few runs, compare with FP32 to ensure convergence

## Example: Training with BF16

```bash
# Standard SlotContrast with BF16
torchrun --nproc_per_node=8 -m slotcontrast.train \
    configs/slotcontrast/movi_e.yaml \
    --bf16 \
    --use-optimizations

# STEVE with BF16 (low memory)
torchrun --nproc_per_node=8 -m slotcontrast.train \
    configs/integrated_v2_movi_e_lowmem.yaml \
    --bf16 \
    globals.BATCH_SIZE_PER_GPU=4  # Can increase with BF16
```

## Memory Savings Example

**Without BF16** (vocab_size=512):
- Batch size: 2
- Memory usage: ~12 GB per GPU

**With BF16** (vocab_size=512):
- Batch size: 4-6
- Memory usage: ~8 GB per GPU
- 2-3x faster training

## Further Reading

- [PyTorch BF16 Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA A100 BF16 Guide](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
- [BF16 vs FP16 Comparison](https://arxiv.org/abs/1905.12322)
