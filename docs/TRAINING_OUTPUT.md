# Training Output Format

This document describes the training output format and metrics displayed during model training.

## Progress Bar Display

During training, you will see the following metrics in the progress bar for each epoch:

```
Epoch X/Y: 100%|████████| 50/50 [00:05<00:00, 10.00it/s, train_loss=0.234, train_accuracy=0.892, val_loss=0.267, val_accuracy=0.875]
```

### Displayed Metrics

The progress bar shows these **4 key metrics** per epoch:

1. **`train_loss`** - Training loss (averaged over all batches)
2. **`train_accuracy`** - Training accuracy
3. **`val_loss`** - Validation loss (averaged over all batches)
4. **`val_accuracy`** - Validation accuracy

### Additional Metrics (Logger Only)

Other metrics configured in your config file (e.g., `f1score`, `precision`, `recall`) are computed and logged to TensorBoard/WandB but **not shown in the progress bar** to keep the output clean.

## Configuration

Metrics are configured in `configs/default_config.yaml`:

```yaml
training:
  base:
    metrics: ['accuracy', 'f1score', 'precision', 'recall']
```

The first metric should always be `'accuracy'` as it's displayed in the progress bar.

## Example Training Session

```
============================================================
TRAINING
============================================================

Using accelerator: gpu
Training for 50 epochs...
Batch size: 32
Learning rate: 0.001
Optimizer: adam
LR Scheduler: warmup_cosine
  Warmup epochs: 5
  Min LR: 1.0e-06
  Warmup start LR: 1.0e-07

Epoch 1/50:  100%|██████████| 45/45 [00:08<00:00,  5.23it/s, train_loss=1.234, train_accuracy=0.456, val_loss=1.189, val_accuracy=0.478]
Epoch 2/50:  100%|██████████| 45/45 [00:08<00:00,  5.31it/s, train_loss=0.987, train_accuracy=0.612, val_loss=0.965, val_accuracy=0.625]
Epoch 3/50:  100%|██████████| 45/45 [00:08<00:00,  5.28it/s, train_loss=0.756, train_accuracy=0.734, val_loss=0.789, val_accuracy=0.721]
...
Epoch 48/50: 100%|██████████| 45/45 [00:08<00:00,  5.29it/s, train_loss=0.089, train_accuracy=0.967, val_loss=0.124, val_accuracy=0.953]
Epoch 49/50: 100%|██████████| 45/45 [00:08<00:00,  5.30it/s, train_loss=0.087, train_accuracy=0.969, val_loss=0.121, val_accuracy=0.955]
Epoch 50/50: 100%|██████████| 45/45 [00:08<00:00,  5.27it/s, train_loss=0.085, train_accuracy=0.971, val_loss=0.119, val_accuracy=0.957]

============================================================
Base model saved to: checkpoints/ecog_atcnet_base_model.pt
============================================================
```

## Logging to Files

All metrics (including those not shown in progress bar) are logged to:

1. **TensorBoard logs** - Located in `logs/` directory
   - View with: `tensorboard --logdir logs/`

2. **CSV logs** - PyTorch Lightning creates CSV logs automatically
   - Located in `logs/lightning_logs/version_X/metrics.csv`

## Viewing All Metrics

To view all metrics including F1, precision, recall:

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

Or read the CSV file:

```python
import pandas as pd
metrics = pd.read_csv('logs/lightning_logs/version_0/metrics.csv')
print(metrics[['epoch', 'train_loss', 'val_loss', 'train_accuracy',
               'val_accuracy', 'train_f1score', 'val_f1score']])
```

## Customizing Display

To change which metric is shown in the progress bar, modify `utils/trainer.py`:

```python
# In on_train_epoch_end() and on_validation_epoch_end()
show_in_prog_bar = (metric_name == 'accuracy')  # Change 'accuracy' to your preferred metric
```

For example, to show F1-score instead:

```python
show_in_prog_bar = (metric_name == 'f1score')
```

## Best Practices

1. **Monitor both train and val metrics** - Large gap indicates overfitting
2. **Watch val_loss** - Should decrease; if it increases, you may be overfitting
3. **Check val_accuracy** - Primary metric for classification tasks
4. **Use TensorBoard** - For detailed analysis and comparing runs
5. **Early stopping** - Stop if val_loss stops improving for several epochs

## Troubleshooting

### No metrics shown

- Check `verbose=True` in trainer configuration
- Verify metrics are in config: `training.base.metrics`

### Metrics are NaN

- Check learning rate (might be too high)
- Verify data preprocessing is correct
- Check for numerical instability

### Progress bar too cluttered

- Only 4 metrics are shown by default
- Other metrics are logged but not displayed
- This is intentional for clean output
