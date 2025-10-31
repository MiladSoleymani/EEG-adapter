# Quick Start Guide

This guide will help you get started with the ECoG-ATCNet-LoRA project in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure Your Data

Edit `configs/default_config.yaml` and update the data path:

```yaml
data:
  data_prefix: "/path/to/your/data"  # <-- Change this
```

Also update the subject/session lists:

```yaml
data:
  # Base training (multiple subjects)
  base_train_subjects: [2, 3, 4, 5]
  base_train_sessions: [0, 1]

  # Validation
  base_val_subjects: [1]
  base_val_sessions: [1]

  # LoRA fine-tuning (target subject)
  lora_train_subjects: [1]
  lora_train_sessions: [0]
```

## Step 3: Train Base Model

```bash
python scripts/train_base.py
```

This will:
- Load data from multiple subjects
- Train ATCNet model
- Save to `checkpoints/ecog_atcnet_base_model.pt`

## Step 4: Fine-tune with LoRA

```bash
python scripts/train_lora.py
```

This will:
- Load the base model
- Add LoRA adapters (only ~1% parameters trainable)
- Fine-tune on target subject
- Save to `checkpoints/ecog_atcnet_lora_model.pt`

## Step 5: Evaluate

```bash
python scripts/evaluate.py
```

This will compare base vs. LoRA fine-tuned performance.

## Customizing LoRA Configuration

### Quick Method: Use the Config Generator

```bash
# Example 1: Change LoRA rank to 16
python scripts/create_custom_config.py --rank 16 --alpha 32

# Example 2: Target attention layers
python scripts/create_custom_config.py --target-modules msa

# Example 3: Higher learning rate
python scripts/create_custom_config.py --lr 0.001

# Example 4: Full customization
python scripts/create_custom_config.py \
  --rank 32 \
  --alpha 64 \
  --lr 0.001 \
  --target-modules dense,msa \
  --output my_experiment
```

Then train with your custom config:

```bash
python scripts/train_lora.py --config configs/my_experiment.yaml
```

### Manual Method: Edit YAML File

Edit `configs/default_config.yaml`:

```yaml
# LoRA Configuration
lora:
  enabled: true
  rank: 16              # Try: 4, 8, 16, 32
  alpha: 32             # Usually 2 × rank
  target_modules: ['dense', 'msa']  # Options: dense, msa, tcn
  dropout: 0.0

# Training Configuration - LoRA
training:
  lora:
    batch_size: 32
    max_epochs: 30
    learning_rate: 0.001  # Try: 0.0001, 0.0005, 0.001
    weight_decay: 0.0
    optimizer: 'adam'
```

## Common LoRA Configurations

### Conservative (Small data, avoid overfitting)
```yaml
lora:
  rank: 4
  alpha: 8
  target_modules: ['dense']
training:
  lora:
    learning_rate: 0.0001
    max_epochs: 20
```

### Balanced (Default, good starting point)
```yaml
lora:
  rank: 8
  alpha: 16
  target_modules: ['dense']
training:
  lora:
    learning_rate: 0.0005
    max_epochs: 30
```

### Aggressive (More data available)
```yaml
lora:
  rank: 16
  alpha: 32
  target_modules: ['dense', 'msa']
training:
  lora:
    learning_rate: 0.001
    max_epochs: 50
```

### Full Model (All layers)
```yaml
lora:
  rank: 32
  alpha: 64
  target_modules: ['dense', 'msa', 'tcn']
training:
  lora:
    learning_rate: 0.001
    max_epochs: 50
```

## Hyperparameter Tuning Tips

### LoRA Rank
- **Rank 4**: Very parameter-efficient, may underfit
- **Rank 8**: Good default, balanced performance
- **Rank 16**: More capacity, better for complex adaptations
- **Rank 32**: Maximum capacity, risk of overfitting on small data

### LoRA Alpha
- Usually set to `2 × rank`
- Higher alpha = stronger LoRA influence
- Lower alpha = more conservative adaptation

### Learning Rate
- **0.0001**: Conservative, slower convergence
- **0.0005**: Good default for LoRA
- **0.001**: Aggressive, faster convergence (may be unstable)

### Target Modules
- **['dense']**: Only classification heads (most efficient)
- **['dense', 'msa']**: Add attention adaptation
- **['dense', 'msa', 'tcn']**: Full model adaptation

## Experiment Tracking

Create multiple configs and compare:

```bash
# Experiment 1: Small rank
python scripts/create_custom_config.py --rank 4 --alpha 8 --output exp1_small
python scripts/train_lora.py --config configs/exp1_small.yaml

# Experiment 2: Large rank
python scripts/create_custom_config.py --rank 32 --alpha 64 --output exp2_large
python scripts/train_lora.py --config configs/exp2_large.yaml

# Experiment 3: Different modules
python scripts/create_custom_config.py --rank 8 --target-modules dense,msa --output exp3_modules
python scripts/train_lora.py --config configs/exp3_modules.yaml

# Compare all
python scripts/evaluate.py
```

## Troubleshooting

### "No data loaded" error
- Check `data_prefix` path in config
- Verify .fif files exist
- Check subject/session IDs match your data structure

### Out of memory
```yaml
training:
  lora:
    batch_size: 16  # Reduce from 32
```

### Poor performance
- Increase LoRA rank: 8 → 16
- Increase learning rate: 0.0005 → 0.001
- Add more target modules: ['dense'] → ['dense', 'msa']
- Train longer: max_epochs: 30 → 50

### Model not improving
- Decrease learning rate: 0.001 → 0.0005
- Decrease LoRA rank: 16 → 8
- Check data preprocessing (freq_band, time_range)

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check the [notebook](notebooks/ecog-atcnet-lora.ipynb) for interactive examples
3. Experiment with different LoRA configurations
4. Try different target subjects/sessions
5. Visualize results using the utils module

## Getting Help

- Open an issue on GitHub
- Check the troubleshooting section in README.md
- Review the configuration comments in `configs/default_config.yaml`
