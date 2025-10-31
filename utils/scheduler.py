"""Learning rate scheduler utilities for training."""

import math
import warnings
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine annealing.

    During warmup, learning rate increases linearly from warmup_start_lr to base_lr.
    After warmup, learning rate decreases following a cosine curve to min_lr.

    This is commonly used in transformer training and has shown good results
    for various deep learning tasks.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize WarmupCosineScheduler.

        Parameters
        ----------
        optimizer : Optimizer
            Wrapped optimizer
        warmup_epochs : int
            Number of epochs for linear warmup
        max_epochs : int
            Total number of training epochs
        warmup_start_lr : float
            Starting learning rate for warmup phase
        min_lr : float
            Minimum learning rate for cosine annealing
        last_epoch : int
            The index of last epoch
        verbose : bool
            If True, prints a message to stdout for each update
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr

        if warmup_epochs >= max_epochs:
            warnings.warn(
                f"warmup_epochs ({warmup_epochs}) >= max_epochs ({max_epochs}). "
                "Warmup will cover entire training."
            )

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        list of float
            Learning rates for each parameter group
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                self._warmup_lr(self.last_epoch, base_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            return [
                self._cosine_annealing_lr(self.last_epoch, base_lr)
                for base_lr in self.base_lrs
            ]

    def _warmup_lr(self, epoch: int, base_lr: float) -> float:
        """
        Compute learning rate during warmup phase.

        Parameters
        ----------
        epoch : int
            Current epoch
        base_lr : float
            Base learning rate for this parameter group

        Returns
        -------
        float
            Learning rate for current epoch
        """
        if self.warmup_epochs == 0:
            return base_lr

        # Linear interpolation from warmup_start_lr to base_lr
        return self.warmup_start_lr + \
            (base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs

    def _cosine_annealing_lr(self, epoch: int, base_lr: float) -> float:
        """
        Compute learning rate during cosine annealing phase.

        Parameters
        ----------
        epoch : int
            Current epoch
        base_lr : float
            Base learning rate for this parameter group

        Returns
        -------
        float
            Learning rate for current epoch
        """
        # Cosine annealing from base_lr to min_lr
        progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)  # Clamp to [0, 1]

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr - self.min_lr) * cosine_decay


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    max_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    warmup_start_lr: Optional[float] = None,
    step_size: int = 30,
    gamma: float = 0.1,
    verbose: bool = False
) -> Optional[_LRScheduler]:
    """
    Create a learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule
    scheduler_type : str
        Type of scheduler: 'warmup_cosine', 'cosine', 'step', 'exponential', 'none'
    max_epochs : int
        Maximum number of training epochs
    warmup_epochs : int
        Number of warmup epochs (for warmup schedulers)
    min_lr : float
        Minimum learning rate (for cosine schedulers)
    warmup_start_lr : float, optional
        Starting LR for warmup. If None, uses min_lr
    step_size : int
        Period of learning rate decay for StepLR
    gamma : float
        Multiplicative factor of learning rate decay
    verbose : bool
        If True, prints a message for each update

    Returns
    -------
    _LRScheduler or None
        Learning rate scheduler, or None if scheduler_type is 'none'
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'none':
        return None

    elif scheduler_type == 'warmup_cosine':
        if warmup_start_lr is None:
            warmup_start_lr = min_lr

        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            min_lr=min_lr,
            verbose=verbose
        )

    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max_epochs,
            eta_min=min_lr,
            verbose=verbose
        )

    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
            verbose=verbose
        )

    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=gamma,
            verbose=verbose
        )

    else:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type}. "
            f"Supported types: warmup_cosine, cosine, step, exponential, none"
        )
