"""Script to create custom configuration files with different LoRA settings."""

import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    lr: float = 0.0005,
    target_modules: str = "dense",
    output_name: str = None
):
    """
    Create a custom config file with specified LoRA settings.

    Parameters
    ----------
    rank : int
        LoRA rank
    alpha : int
        LoRA alpha
    lr : float
        Learning rate for LoRA training
    target_modules : str
        Comma-separated list of target modules
    output_name : str
        Output filename (without .yaml extension)
    """
    # Load default config
    config = load_config()

    # Update LoRA settings
    config.set('lora.rank', rank)
    config.set('lora.alpha', alpha)
    config.set('training.lora.learning_rate', lr)

    # Parse target modules
    modules = [m.strip() for m in target_modules.split(',')]
    config.set('lora.target_modules', modules)

    # Generate output filename
    if output_name is None:
        modules_str = "_".join(modules)
        output_name = f"lora_r{rank}_a{alpha}_lr{lr}_{modules_str}"

    output_path = f"configs/{output_name}.yaml"

    # Save config
    config.save(output_path)

    print("=" * 60)
    print("CUSTOM CONFIGURATION CREATED")
    print("=" * 60)
    print(f"\nConfiguration saved to: {output_path}")
    print("\nSettings:")
    print(f"  LoRA Rank:         {rank}")
    print(f"  LoRA Alpha:        {alpha}")
    print(f"  Learning Rate:     {lr}")
    print(f"  Target Modules:    {modules}")
    print("\nTo use this configuration:")
    print(f"  python scripts/train_lora.py --config {output_path}")
    print("=" * 60)


def main():
    """Parse arguments and create custom config."""
    parser = argparse.ArgumentParser(
        description="Create custom LoRA configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create config with rank=16, alpha=32
  python scripts/create_custom_config.py --rank 16 --alpha 32

  # Create config targeting attention layers
  python scripts/create_custom_config.py --target-modules msa

  # Create config with custom learning rate
  python scripts/create_custom_config.py --lr 0.001

  # Full customization
  python scripts/create_custom_config.py --rank 32 --alpha 64 --lr 0.001 --target-modules dense,msa --output my_config
        """
    )

    parser.add_argument(
        '--rank',
        type=int,
        default=8,
        help='LoRA rank (default: 8). Try 4, 8, 16, or 32.'
    )
    parser.add_argument(
        '--alpha',
        type=int,
        default=16,
        help='LoRA alpha (default: 16). Typically 2*rank.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0005,
        help='Learning rate for LoRA training (default: 0.0005)'
    )
    parser.add_argument(
        '--target-modules',
        type=str,
        default='dense',
        help='Comma-separated list of target modules (default: dense). Options: dense, msa, tcn'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename without .yaml extension (default: auto-generated)'
    )

    args = parser.parse_args()

    create_lora_config(
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        target_modules=args.target_modules,
        output_name=args.output
    )


if __name__ == "__main__":
    main()
