"""Script to evaluate trained models."""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config
from data.dataset import MultiSubjectECoGDataset, create_file_list
from models import ATCNet, add_lora_to_model
from utils.trainer import ClassifierTrainer


def evaluate_model(
    model_path: str,
    test_loader: DataLoader,
    num_classes: int,
    accelerator: str = 'cpu',
    model_name: str = "Model",
    is_lora: bool = False
):
    """
    Evaluate a trained model.

    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    test_loader : DataLoader
        Test data loader
    num_classes : int
        Number of classes
    accelerator : str
        Device to use
    model_name : str
        Name for display
    is_lora : bool
        Whether the model uses LoRA adapters

    Returns
    -------
    dict
        Test results
    """
    # Get model architecture from test data
    sample_batch = next(iter(test_loader))
    sample_shape = sample_batch[0].shape
    num_channels = sample_shape[2]
    chunk_size = sample_shape[3]

    # Load config to get model architecture
    config = load_config()

    # Initialize model
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

    # Add LoRA adapters if needed
    if is_lora:
        lora_config = config.get('lora')
        if lora_config.get('enabled', False):
            model = add_lora_to_model(
                model,
                rank=lora_config.get('rank', 8),
                alpha=lora_config.get('alpha', 16),
                target_modules=lora_config.get('target_modules', ['dense'])
            )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"\n{model_name} loaded from: {model_path}")

    # Initialize trainer
    trainer = ClassifierTrainer(
        model=model,
        num_classes=num_classes,
        accelerator=accelerator,
        metrics=['accuracy', 'f1score', 'precision', 'recall']
    )

    # Evaluate
    print(f"\nEvaluating {model_name}...")
    results = trainer.test(test_loader)

    return results[0]


def main(config_path=None, base_only=False, lora_only=False):
    """
    Evaluate trained models.

    Parameters
    ----------
    config_path : str, optional
        Path to custom configuration file
    base_only : bool
        Only evaluate base model
    lora_only : bool
        Only evaluate LoRA model
    """
    # Load configuration
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    config = load_config(config_path)

    # Set random seed
    torch.manual_seed(config.seed)

    # Create test file list
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    test_files = create_file_list(
        data_prefix=config.get('data.data_prefix'),
        subject_ids=config.get('data.test_subjects'),
        session_ids=config.get('data.test_sessions')
    )
    print(f"\nTest files ({len(test_files)} files):")
    for f in test_files:
        print(f"  - {f}")

    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = MultiSubjectECoGDataset(
        file_list=test_files,
        freq_band=config.get('data.freq_band'),
        time_range=config.get('data.time_range'),
        remove_bad=config.get('data.remove_bad_channels'),
        scale=config.get('data.scale')
    )

    num_classes = test_dataset.num_classes
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")

    # Create data loader
    batch_size = config.get('training.base.batch_size')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('hardware.num_workers')
    )

    # Determine accelerator
    accelerator = config.get('hardware.accelerator')
    if accelerator == 'auto':
        if torch.cuda.is_available():
            accelerator = 'gpu'
        elif torch.backends.mps.is_available():
            accelerator = 'mps'
        else:
            accelerator = 'cpu'

    print(f"Using accelerator: {accelerator}")

    # Evaluate models
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    results = {}

    # Evaluate base model
    if not lora_only:
        base_model_path = config.get('paths.base_model_path')
        if os.path.exists(base_model_path):
            base_results = evaluate_model(
                model_path=base_model_path,
                test_loader=test_loader,
                num_classes=num_classes,
                accelerator=accelerator,
                model_name="Base Model"
            )
            results['base'] = base_results
        else:
            print(f"\nBase model not found at: {base_model_path}")
            print("Skipping base model evaluation.")

    # Evaluate LoRA model
    if not base_only:
        lora_model_path = config.get('paths.lora_model_path')
        if os.path.exists(lora_model_path):
            lora_results = evaluate_model(
                model_path=lora_model_path,
                test_loader=test_loader,
                num_classes=num_classes,
                accelerator=accelerator,
                model_name="LoRA Fine-tuned Model",
                is_lora=True  # Enable LoRA adapters for loading
            )
            results['lora'] = lora_results
        else:
            print(f"\nLoRA model not found at: {lora_model_path}")
            print("Skipping LoRA model evaluation.")

    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)

        # Extract metrics
        metrics = [k for k in results['base'].keys() if k != 'test_loss']

        print(f"\n{'Metric':<20} {'Base Model':<15} {'LoRA Model':<15} {'Improvement':<15}")
        print("-" * 65)

        for metric in metrics:
            base_val = results['base'].get(metric, 0)
            lora_val = results['lora'].get(metric, 0)
            improvement = lora_val - base_val

            print(f"{metric:<20} {base_val:<15.4f} {lora_val:<15.4f} {improvement:+.4f}")

        # Overall summary
        base_acc = results['base'].get('test_accuracy', 0)
        lora_acc = results['lora'].get('test_accuracy', 0)
        acc_improvement = lora_acc - base_acc

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Base Model Accuracy:        {base_acc:.2%}")
        print(f"LoRA Fine-tuned Accuracy:   {lora_acc:.2%}")
        print(f"Improvement:                {acc_improvement:+.2%}")

        if acc_improvement > 0:
            print(f"\nLoRA fine-tuning improved accuracy by {acc_improvement:.2%}!")
        elif acc_improvement < 0:
            print(f"\nLoRA fine-tuning decreased accuracy by {abs(acc_improvement):.2%}")
        else:
            print("\nNo change in accuracy after LoRA fine-tuning")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file'
    )
    parser.add_argument(
        '--base-only',
        action='store_true',
        help='Only evaluate base model'
    )
    parser.add_argument(
        '--lora-only',
        action='store_true',
        help='Only evaluate LoRA model'
    )
    args = parser.parse_args()

    main(args.config, args.base_only, args.lora_only)
