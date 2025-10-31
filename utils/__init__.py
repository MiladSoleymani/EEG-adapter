"""Utilities module for ECoG-ATCNet-LoRA."""

from .visualization import (
    compute_mean_psd,
    plot_mean_psd,
    plot_channel_statistics,
    plot_parameter_comparison,
    plot_training_curves,
    plot_confusion_matrix,
    visualize_sample_signals
)

from .scheduler import (
    WarmupCosineScheduler,
    create_scheduler
)

from .trainer import (
    ClassifierTrainer,
    classification_metrics
)

__all__ = [
    'compute_mean_psd',
    'plot_mean_psd',
    'plot_channel_statistics',
    'plot_parameter_comparison',
    'plot_training_curves',
    'plot_confusion_matrix',
    'visualize_sample_signals',
    'WarmupCosineScheduler',
    'create_scheduler',
    'ClassifierTrainer',
    'classification_metrics'
]
