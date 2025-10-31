# LoRA Target Modules Guide

This guide explains what you can put in the `target_modules` configuration for LoRA fine-tuning.

## Overview

The `target_modules` parameter controls which layers in the model get LoRA adapters. It accepts a list of substring patterns that match layer names.

## Available Options for ATCNet

### Option 1: Classification Heads Only (Conservative)

```yaml
lora:
  target_modules: ['dense']
```

**What it targets:**
- `dense0` - Classification head for window 0
- `dense1` - Classification head for window 1
- `dense2` - Classification head for window 2

**Total:** 3 LoRA adapters

**Use when:**
- You want minimal parameter changes
- Dataset is very small (< 100 samples)
- Quick experimentation

**Trainable params:** ~1,200 (1.5% of model)

---

### Option 2: Attention Output Projections Only

```yaml
lora:
  target_modules: ['msa']
```

**What it targets:**
- `msa0.out_proj` - Attention output projection for window 0
- `msa1.out_proj` - Attention output projection for window 1
- `msa2.out_proj` - Attention output projection for window 2

**Total:** 3 LoRA adapters

**Use when:**
- You want to adapt feature representations
- Focus on attention mechanism adaptation

**Trainable params:** ~1,200 (1.5% of model)

---

### Option 3: Both (Recommended) ⭐

```yaml
lora:
  target_modules: ['dense', 'msa']
```

**What it targets:**
- All 3 classification heads (`dense0`, `dense1`, `dense2`)
- All 3 attention output projections (`msa0.out_proj`, `msa1.out_proj`, `msa2.out_proj`)

**Total:** 6 LoRA adapters

**Use when:**
- Standard fine-tuning scenario
- Want to adapt both features and classification
- Recommended for most use cases

**Trainable params:** ~2,400 (3% of model)

**Benefits:**
- ✓ Adapts both attention mechanism and classification
- ✓ Better performance than single target
- ✓ Still parameter-efficient (97% reduction vs full fine-tuning)

---

## Complete Configuration Example

```yaml
lora:
  enabled: true
  rank: 8  # Low-rank dimension
  alpha: 16  # Scaling factor
  target_modules: ['dense', 'msa']  # Recommended
  dropout: 0.1
```

## Advanced: Custom Patterns

You can target any layer by name pattern:

```yaml
# Target all linear layers (not recommended - too many)
target_modules: ['']  # Empty string matches everything

# Target only first window's modules
target_modules: ['dense0', 'msa0']

# Target dense and a specific attention layer
target_modules: ['dense', 'msa2']
```

## How It Works

The LoRA implementation searches for layer names containing the specified substrings:

```python
for name, module in model.named_modules():
    if any(target in name for target in target_modules):
        if isinstance(module, nn.Linear):
            # Add LoRA adapter
            ...
```

## Available Linear Layers in ATCNet

Run training with verbose output to see all available layers:

```
Available module names:
  - dense0
  - dense1
  - dense2
  - msa0.in_proj_weight
  - msa0.out_proj
  - msa1.in_proj_weight
  - msa1.out_proj
  - msa2.in_proj_weight
  - msa2.out_proj
```

**Note:** Only `out_proj` is targeted for `msa` because `in_proj_weight` is not a standard `nn.Linear` layer.

## Parameter Comparison

| Configuration | LoRA Adapters | Trainable Params | % of Model |
|--------------|---------------|------------------|------------|
| `['dense']` | 3 | ~1,200 | 1.5% |
| `['msa']` | 3 | ~1,200 | 1.5% |
| `['dense', 'msa']` | 6 | ~2,400 | 3.0% |
| Full fine-tuning | N/A | ~78,000 | 100% |

## Recommendations by Dataset Size

| Dataset Size | Recommended Config | Reasoning |
|-------------|-------------------|-----------|
| < 50 samples | `['dense']` | Minimal adaptation, avoid overfitting |
| 50-200 samples | `['dense', 'msa']` | Standard LoRA, good balance |
| > 200 samples | `['dense', 'msa']` with higher rank | More capacity for larger datasets |

## Tuning LoRA Parameters

If you want more/less adaptation capacity:

```yaml
# More capacity (rank 16, alpha 32)
lora:
  rank: 16
  alpha: 32
  target_modules: ['dense', 'msa']

# Less capacity (rank 4, alpha 8)
lora:
  rank: 4
  alpha: 8
  target_modules: ['dense']
```

## Troubleshooting

### No layers matched

```
Warning: No modules matched target_modules=['xyz']
```

**Solution:** Check the spelling and use `['dense']` or `['msa']` for ATCNet.

### Too many parameters

If training is slow or overfitting:
1. Reduce `rank` (e.g., from 8 to 4)
2. Use fewer targets (e.g., only `['dense']`)
3. Increase `dropout`

### Underfitting

If performance is poor:
1. Increase `rank` (e.g., from 8 to 16)
2. Add more targets (e.g., `['dense', 'msa']`)
3. Decrease `dropout`

## Technical Details

### What is LoRA?

LoRA (Low-Rank Adaptation) freezes the original model weights and adds small trainable matrices:

```
Output = W_original(x) + (B @ A)(x) * scaling
```

Where:
- `W_original` is frozen
- `A` and `B` are trainable (rank × input_dim and output_dim × rank)
- `scaling = alpha / rank`

### Why Multi-Head Attention Works Now

Previously, applying LoRA to `MultiheadAttention.out_proj` failed because PyTorch expected direct `weight` attribute access.

**Fix:** Added `@property` decorators to `LinearWithLoRA`:

```python
@property
def weight(self):
    return self.linear.weight

@property
def bias(self):
    return self.linear.bias
```

This makes the wrapper fully compatible with any PyTorch module that expects standard `nn.Linear` attribute access patterns.

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original LoRA publication
- [ATCNet Paper](https://arxiv.org/abs/2106.11170) - Model architecture
- Project docs: `docs/QUICKSTART.md`
