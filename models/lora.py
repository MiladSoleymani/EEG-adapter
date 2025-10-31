"""Low-Rank Adaptation (LoRA) implementation for efficient fine-tuning."""

import math
import torch
import torch.nn as nn
from typing import List


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.

    Adds trainable low-rank matrices A and B to approximate weight updates
    without modifying the original weights.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: int = 16):
        """
        Initialize LoRA layer.

        Parameters
        ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        rank : int
            Rank of the low-rank decomposition
        alpha : int
            Scaling factor for LoRA updates
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            LoRA adjustment: (x @ A^T @ B^T) * scaling
        """
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps a frozen linear layer with a trainable LoRA adapter.
    Output = Linear(x) + LoRA(x)
    """

    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        """
        Initialize Linear layer with LoRA.

        Parameters
        ----------
        linear_layer : nn.Linear
            Original linear layer (will be frozen)
        rank : int
            LoRA rank
        alpha : int
            LoRA alpha scaling factor
        """
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

        # Store dimensions for compatibility
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Freeze original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        """Expose weight attribute for compatibility with PyTorch modules."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose bias attribute for compatibility with PyTorch modules."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Linear(x) + LoRA(x)
        """
        return self.linear(x) + self.lora(x)


def add_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: List[str] = ['dense']
) -> nn.Module:
    """
    Add LoRA adapters to specified modules in a model.

    Parameters
    ----------
    model : nn.Module
        Model to add LoRA to
    rank : int
        LoRA rank
    alpha : int
        LoRA alpha scaling factor
    target_modules : list of str
        List of module name substrings to target for LoRA.
        E.g., ['dense', 'linear', 'fc'] will add LoRA to all modules
        whose names contain these substrings.

        For ATCNet:
        - ['dense'] targets classification heads (3 layers)
        - ['msa'] targets multi-head attention output projections (3 layers)
        - ['dense', 'msa'] targets both (6 layers total, recommended)

    Returns
    -------
    nn.Module
        Model with LoRA adapters added
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Find and replace target modules with LoRA versions
    modified_count = 0
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        should_add_lora = any(target in name for target in target_modules)

        if should_add_lora and isinstance(module, nn.Linear):
            # Get parent module and child name
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRA version
            lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            print(f"Added LoRA to: {name}")
            modified_count += 1

    if modified_count == 0:
        print(f"Warning: No modules matched target_modules={target_modules}")
        print("Available module names:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  - {name}")

    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters in a model.

    Parameters
    ----------
    model : nn.Module
        Model to count parameters for

    Returns
    -------
    trainable : int
        Number of trainable parameters
    total : int
        Total number of parameters
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from a model.

    Parameters
    ----------
    model : nn.Module
        Model containing LoRA layers

    Returns
    -------
    list of nn.Parameter
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model weights.

    This permanently applies the LoRA adaptations to the original weights,
    removing the LoRA layers. Useful for deployment.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA adapters

    Returns
    -------
    nn.Module
        Model with merged weights (LoRA removed)
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # Compute merged weight
            with torch.no_grad():
                lora_weight = (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling
                module.linear.weight.data += lora_weight

                # Unfreeze the merged weights
                module.linear.weight.requires_grad = True

            # Get parent and replace module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with the original linear layer (now with merged weights)
            setattr(parent, child_name, module.linear)
            print(f"Merged LoRA weights into: {name}")

    return model


def print_lora_info(model: nn.Module):
    """
    Print information about LoRA adapters in a model.

    Parameters
    ----------
    model : nn.Module
        Model to analyze
    """
    trainable, total = count_parameters(model)
    reduction = 100 * (1 - trainable / total)

    print("\n" + "=" * 60)
    print("LoRA Configuration Summary")
    print("=" * 60)
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Frozen parameters:     {total - trainable:,}")
    print(f"Trainable percentage:  {100 * trainable / total:.2f}%")
    print(f"Parameter reduction:   {reduction:.2f}%")

    # Count LoRA layers
    lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALayer))
    print(f"\nNumber of LoRA adapters: {lora_count}")

    # Show LoRA layer details
    if lora_count > 0:
        print("\nLoRA layers:")
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRA):
                print(f"  {name}: rank={module.lora.rank}, alpha={module.lora.alpha}")

    print("=" * 60 + "\n")
