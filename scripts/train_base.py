"""Script to train the base ATCNet model on multiple subjects."""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config
from data.dataset import MultiSubjectECoGDataset, create_file_list
from models import ATCNet, count_parameters
from trainer import ClassifierTrainer


def main(config_path=None):
    """
    Train base ATCNet model.

    Parameters
    ----------
    config_path : str, optional
        Path to custom configuration file
    """
    # Load configuration
    print("=" * 60)
    print("BASE MODEL TRAINING")
    print("=" * 60)

    config = load_config(config_path)
    print("\nLoaded configuration:")
    print(config)

    # Set random seed
    torch.manual_seed(config.seed)

    # Create file lists
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_files = create_file_list(
        data_prefix=config.get('data.data_prefix'),
        subject_ids=config.get('data.base_train_subjects'),
        session_ids=config.get('data.base_train_sessions')
    )
    print(f"\nBase training files ({len(train_files)} files):")
    for f in train_files:
        print(f"  - {f}")

    val_files = create_file_list(
        data_prefix=config.get('data.data_prefix'),
        subject_ids=config.get('data.base_val_subjects'),
        session_ids=config.get('data.base_val_sessions')
    )
    print(f"\nBase validation files ({len(val_files)} files):")
    for f in val_files:
        print(f"  - {f}")

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = MultiSubjectECoGDataset(
        file_list=train_files,
        freq_band=config.get('data.freq_band'),
        time_range=config.get('data.time_range'),
        remove_bad=config.get('data.remove_bad_channels'),
        scale=config.get('data.scale')
    )

    print("\nCreating validation dataset...")
    val_dataset = MultiSubjectECoGDataset(
        file_list=val_files,
        freq_band=config.get('data.freq_band'),
        time_range=config.get('data.time_range'),
        remove_bad=config.get('data.remove_bad_channels'),
        scale=config.get('data.scale')
    )

    # Get dataset info
    num_channels = train_dataset.num_channels
    chunk_size = train_dataset.num_timepoints
    num_classes = train_dataset.num_classes

    print(f"\n" + "=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    print(f"Number of channels:    {num_channels}")
    print(f"Chunk size:            {chunk_size}")
    print(f"Number of classes:     {num_classes}")
    print(f"Training samples:      {len(train_dataset)}")
    print(f"Validation samples:    {len(val_dataset)}")

    # Create data loaders
    batch_size = config.get('training.base.batch_size')
    num_workers = config.get('hardware.num_workers')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize model
    print(f"\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    model = ATCNet(
        in_channels=config.get('model.in_channels'),
        num_classes=num_classes,
        num_windows=config.get('model.num_windows'),
        num_electrodes=num_channels,
        chunk_size=chunk_size,
        F1=config.get('model.F1'),
        D=config.get('model.D'),
        conv_pool_size=config.get('model.conv_pool_size'),
        tcn_kernel_size=config.get('model.tcn_kernel_size'),
        tcn_depth=config.get('model.tcn_depth')
    )

    trainable, total = count_parameters(model)
    print(f"\nBase model parameters:")
    print(f"  Trainable:  {trainable:,}")
    print(f"  Total:      {total:,}")
    print(f"  Percentage: {100 * trainable / total:.2f}%")

    # Initialize trainer
    print(f"\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    accelerator = config.get('hardware.accelerator')
    if accelerator == 'auto':
        if torch.cuda.is_available():
            accelerator = 'gpu'
        elif torch.backends.mps.is_available():
            accelerator = 'mps'
        else:
            accelerator = 'cpu'

    print(f"\nUsing accelerator: {accelerator}")

    trainer = ClassifierTrainer(
        model=model,
        num_classes=num_classes,
        lr=config.get('training.base.learning_rate'),
        weight_decay=config.get('training.base.weight_decay'),
        devices=config.get('hardware.devices'),
        accelerator=accelerator,
        verbose=config.get('logging.verbose'),
        metrics=config.get('training.base.metrics'),
        optimizer=config.get('training.base.optimizer')
    )

    max_epochs = config.get('training.base.max_epochs')
    print(f"Training for {max_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {config.get('training.base.learning_rate')}")
    print(f"Optimizer: {config.get('training.base.optimizer')}")

    # Train
    trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    # Save model
    save_path = config.get('paths.base_model_path')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\n" + "=" * 60)
    print(f"Base model saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train base ATCNet model")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file'
    )
    args = parser.parse_args()

    main(args.config)
