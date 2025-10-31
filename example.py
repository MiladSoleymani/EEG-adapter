"""
Example script demonstrating how to use the ECoG-ATCNet-LoRA library.

This script shows the basic workflow:
1. Loading configuration
2. Creating datasets
3. Building models
4. Training and evaluation
"""

import torch
from torch.utils.data import DataLoader

from configs import load_config
from dataset import MultiSubjectECoGDataset, create_file_list
from models import ATCNet, add_lora_to_model, count_parameters, print_lora_info
from trainer import ClassifierTrainer


def main():
    """Run example workflow."""
    print("=" * 60)
    print("ECoG-ATCNet-LoRA Example")
    print("=" * 60)

    # 1. Load configuration
    print("\n1. Loading configuration...")
    config = load_config()

    # You can modify settings programmatically
    config.set('lora.rank', 8)
    config.set('lora.alpha', 16)
    config.set('training.lora.learning_rate', 0.0005)

    print(f"   LoRA Rank: {config.get('lora.rank')}")
    print(f"   LoRA Alpha: {config.get('lora.alpha')}")
    print(f"   Learning Rate: {config.get('training.lora.learning_rate')}")

    # 2. Create file lists
    print("\n2. Creating file lists...")
    train_files = create_file_list(
        data_prefix=config.get('data.data_prefix'),
        subject_ids=[1],  # Example: Subject 1
        session_ids=[0]   # Example: Session 0
    )
    print(f"   Found {len(train_files)} training files")

    # 3. Create datasets (example - will fail without actual data)
    print("\n3. Creating datasets...")
    print("   (This will fail if data files don't exist - that's expected)")
    try:
        dataset = MultiSubjectECoGDataset(
            file_list=train_files,
            freq_band=config.get('data.freq_band'),
            time_range=config.get('data.time_range'),
            remove_bad=config.get('data.remove_bad_channels'),
            scale=config.get('data.scale')
        )
        print(f"   Dataset created with {len(dataset)} samples")

        # Get dataset info
        num_channels = dataset.num_channels
        chunk_size = dataset.num_timepoints
        num_classes = dataset.num_classes
        print(f"   Channels: {num_channels}, Timepoints: {chunk_size}, Classes: {num_classes}")

    except (FileNotFoundError, ValueError) as e:
        print(f"   Expected error (no data files): {e}")
        print("   Using dummy values for demonstration...")
        num_channels = 64
        chunk_size = 1000
        num_classes = 4

    # 4. Create model
    print("\n4. Creating ATCNet model...")
    model = ATCNet(
        in_channels=1,
        num_classes=num_classes,
        num_windows=config.get('model.num_windows'),
        num_electrodes=num_channels,
        chunk_size=chunk_size,
        F1=config.get('model.F1'),
        D=config.get('model.D')
    )

    trainable, total = count_parameters(model)
    print(f"   Base model parameters:")
    print(f"   - Trainable: {trainable:,}")
    print(f"   - Total: {total:,}")

    # 5. Add LoRA
    print("\n5. Adding LoRA adapters...")
    model = add_lora_to_model(
        model,
        rank=config.get('lora.rank'),
        alpha=config.get('lora.alpha'),
        target_modules=config.get('lora.target_modules')
    )

    print_lora_info(model)

    # 6. Initialize trainer
    print("\n6. Initializing trainer...")
    trainer = ClassifierTrainer(
        model=model,
        num_classes=num_classes,
        lr=config.get('training.lora.learning_rate'),
        metrics=['accuracy'],
        accelerator='cpu'  # Use CPU for example
    )
    print("   Trainer initialized")

    # 7. Training (commented out - requires actual data)
    print("\n7. Training...")
    print("   (Skipped - requires actual dataset)")
    print("   To train, uncomment the following code:")
    print("   ```")
    print("   train_loader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)")
    print("   trainer.fit(train_loader, val_loader, max_epochs=30)")
    print("   ```")

    # 8. Save configuration
    print("\n8. Saving custom configuration...")
    config.save('configs/example_config.yaml')
    print("   Configuration saved to: configs/example_config.yaml")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update data_prefix in configs/default_config.yaml")
    print("2. Run: python scripts/train_base.py")
    print("3. Run: python scripts/train_lora.py")
    print("4. Run: python scripts/evaluate.py")
    print("\nOr read QUICKSTART.md for detailed instructions.")


if __name__ == "__main__":
    main()
