# Learning Rate Scheduler

This document describes the learning rate scheduler implementation for the EEG-adapter project.

## Overview

The project now supports various learning rate scheduling strategies, with a particular focus on **Warmup Cosine Annealing**, which has shown excellent results in modern deep learning, especially for transformer-based models and fine-tuning tasks.

## Supported Schedulers

### 1. Warmup Cosine (`warmup_cosine`) - **Recommended**

A two-phase learning rate schedule:
- **Warmup Phase**: Linear increase from `warmup_start_lr` to `learning_rate`
- **Cosine Annealing Phase**: Smooth cosine decay from `learning_rate` to `min_lr`

This scheduler helps with:
- Stable training in early epochs (warmup)
- Smooth convergence in later epochs (cosine annealing)
- Better generalization compared to constant LR

### 2. Cosine Annealing (`cosine`)

Standard cosine annealing without warmup. LR decreases following a cosine curve from `learning_rate` to `min_lr`.

### 3. Step LR (`step`)

Decreases learning rate by a factor of `gamma` every `step_size` epochs.

### 4. Exponential LR (`exponential`)

Decreases learning rate exponentially by multiplying with `gamma` each epoch.

### 5. None (`none`)

Disables learning rate scheduling (constant LR).

## Configuration

Learning rate scheduler is configured in `configs/default_config.yaml`:

```yaml
training:
  base:
    batch_size: 32
    max_epochs: 50
    learning_rate: 0.001
    weight_decay: 0.0
    optimizer: 'adam'
    metrics: ['accuracy', 'f1score', 'precision', 'recall']

    # Learning rate scheduler
    scheduler:
      enabled: true
      type: 'warmup_cosine'  # warmup_cosine, cosine, step, exponential, none
      warmup_epochs: 5  # Number of warmup epochs
      min_lr: 1.0e-6  # Minimum learning rate for cosine annealing
      warmup_start_lr: 1.0e-7  # Starting LR for warmup (if null, uses min_lr)

  lora:
    # Similar configuration for LoRA fine-tuning
    scheduler:
      enabled: true
      type: 'warmup_cosine'
      warmup_epochs: 3
      min_lr: 1.0e-7
      warmup_start_lr: 1.0e-8
```

## Parameters

### Common Parameters

- **`enabled`** (bool): Enable/disable scheduler
- **`type`** (str): Scheduler type (see above)

### Warmup Cosine Parameters

- **`warmup_epochs`** (int): Number of epochs for linear warmup
- **`min_lr`** (float): Minimum learning rate (end of cosine annealing)
- **`warmup_start_lr`** (float, optional): Starting LR for warmup. If `null`, uses `min_lr`

### Step LR Parameters

- **`step_size`** (int): Period of learning rate decay
- **`gamma`** (float): Multiplicative factor of LR decay

### Exponential LR Parameters

- **`gamma`** (float): Multiplicative factor of LR decay per epoch

## Usage Examples

### Example 1: Base Model Training with Warmup Cosine

```yaml
training:
  base:
    learning_rate: 0.001
    max_epochs: 100
    scheduler:
      enabled: true
      type: 'warmup_cosine'
      warmup_epochs: 10  # 10% of training for warmup
      min_lr: 1.0e-6
      warmup_start_lr: 1.0e-8
```

Learning rate schedule:
- Epochs 0-9: Linear increase from 1e-8 to 1e-3
- Epochs 10-99: Cosine decay from 1e-3 to 1e-6

### Example 2: LoRA Fine-tuning with Shorter Warmup

```yaml
training:
  lora:
    learning_rate: 0.0005
    max_epochs: 30
    scheduler:
      enabled: true
      type: 'warmup_cosine'
      warmup_epochs: 3  # Quick warmup for fine-tuning
      min_lr: 1.0e-7
      warmup_start_lr: null  # Uses min_lr as start
```

### Example 3: Disable Scheduler (Constant LR)

```yaml
training:
  base:
    learning_rate: 0.001
    scheduler:
      enabled: false
      type: 'none'
```

## Best Practices

### Base Model Training

- Use longer warmup (5-10 epochs) for stable initialization
- Set `min_lr` to 1e-6 or 1e-7 to allow fine-grained updates
- Total epochs: 50-100

### LoRA Fine-tuning

- Use shorter warmup (2-5 epochs) since starting from pretrained weights
- Use smaller learning rates overall (0.0001-0.001)
- Set lower `min_lr` (1e-7 or 1e-8) for gentler fine-tuning
- Total epochs: 20-50

## Implementation Details

The scheduler is implemented in `utils/scheduler.py` with the following key components:

1. **`WarmupCosineScheduler`**: PyTorch LR scheduler implementing warmup + cosine annealing
2. **`create_scheduler()`**: Factory function for creating any scheduler type
3. Integration with PyTorch Lightning's `configure_optimizers()`

## Testing

Run the scheduler tests:

```bash
python tests/test_scheduler.py
```

This will verify:
- Warmup phase produces linearly increasing LR
- Cosine phase produces smooth decay
- Factory function creates correct scheduler types
- Edge cases are handled properly

## Visualization

The learning rate can be monitored during training:
- PyTorch Lightning automatically logs LR to tensorboard/wandb
- Check logs under the name `learning_rate`

## References

- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187) - Introduced warmup for ResNets
- [BERT Pre-training](https://arxiv.org/abs/1810.04805) - Popularized warmup + cosine for transformers
- [PyTorch Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

## Troubleshooting

### Learning rate not changing

Check that:
1. `scheduler.enabled` is `true` in config
2. `max_epochs` matches your training epochs
3. Scheduler is being stepped each epoch (automatic in PyTorch Lightning)

### Training unstable

Try:
1. Increase `warmup_epochs`
2. Lower `warmup_start_lr`
3. Use a gentler scheduler like `cosine` without warmup

### No improvement over constant LR

- Some tasks may not benefit from scheduling
- Try disabling scheduler or adjusting parameters
- Ensure `min_lr` is not too low (can cause underfitting)
