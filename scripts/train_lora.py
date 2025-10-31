"""Script to fine-tune ATCNet model using LoRA on target subject."""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config
from data.dataset import MultiSubjectECoGDataset, create_file_list
from models import ATCNet, add_lora_to_model, print_lora_info
from trainer import ClassifierTrainer


def main(config_path=None):
    """
    Fine-tune ATCNet model using LoRA.

    Parameters
    ----------
    config_path : str, optional
        Path to custom configuration file
    """
    # Load configuration
    print("=" * 60)
    print("LoRA FINE-TUNING")
    print("=" * 60)

    config = load_config(config_path)
    print("\nLoaded configuration:")
    print(config)

    # Set random seed
    torch.manual_seed(config.seed)

    # Check if base model exists
    base_model_path = config.get('paths.base_model_path')
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. "
            "Please train base model first using train_base.py"
        )

    # Create file lists
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_files = create_file_list(
        data_prefix=config.get('data.data_prefix'),
        subject_ids=config.get('data.lora_train_subjects'),
        session_ids=config.get('data.lora_train_sessions')
    )
    print(f"\nLoRA training files ({len(train_files)} files):")
    for f in train_files:
        print(f"  - {f}")

    val_files = create_file_list(
        data_prefix=config.get('data.data_prefix'),
        subject_ids=config.get('data.base_val_subjects'),
        session_ids=config.get('data.base_val_sessions')
    )
    print(f"\nLoRA validation files ({len(val_files)} files):")
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
    batch_size = config.get('training.lora.batch_size')
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

    # Load base model
    print(f"\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    print(f"\nLoading base model from: {base_model_path}")

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

    model.load_state_dict(torch.load(base_model_path))
    print("Base model loaded successfully")

    # Add LoRA adapters
    if config.get('lora.enabled'):
        print(f"\n" + "=" * 60)
        print("ADDING LoRA ADAPTERS")
        print("=" * 60)

        lora_rank = config.get('lora.rank')
        lora_alpha = config.get('lora.alpha')
        target_modules = config.get('lora.target_modules')

        print(f"\nLoRA configuration:")
        print(f"  Rank:           {lora_rank}")
        print(f"  Alpha:          {lora_alpha}")
        print(f"  Target modules: {target_modules}")

        model = add_lora_to_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=target_modules
        )

        print_lora_info(model)
    else:
        print("\nLoRA is disabled in config. Performing full fine-tuning.")

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
        lr=config.get('training.lora.learning_rate'),
        weight_decay=config.get('training.lora.weight_decay'),
        devices=config.get('hardware.devices'),
        accelerator=accelerator,
        verbose=config.get('logging.verbose'),
        metrics=config.get('training.lora.metrics'),
        optimizer=config.get('training.lora.optimizer')
    )

    max_epochs = config.get('training.lora.max_epochs')
    print(f"Training for {max_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {config.get('training.lora.learning_rate')}")
    print(f"Optimizer: {config.get('training.lora.optimizer')}")

    # Train
    trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    # Save model
    save_path = config.get('paths.lora_model_path')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\n" + "=" * 60)
    print(f"LoRA fine-tuned model saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ATCNet with LoRA")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file'
    )
    args = parser.parse_args()

    main(args.config)
