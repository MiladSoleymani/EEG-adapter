"""Visualization utilities for ECoG/EEG data and model results."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import torch
from typing import List, Tuple, Optional


def compute_mean_psd(
    batch: torch.Tensor,
    fs: int = 250,
    nperseg: int = 256
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute the mean Power Spectral Density for each channel across the batch.

    Parameters
    ----------
    batch : torch.Tensor
        Input batch of shape (batch_size, 1, num_channels, segment_length)
    fs : int
        Sampling frequency in Hz
    nperseg : int
        Length of each segment for Welch's method

    Returns
    -------
    freqs : np.ndarray
        Array of frequency values
    mean_psd_per_channel : list of np.ndarray
        Mean PSD for each channel
    """
    batch_size, _, num_channels, segment_length = batch.shape
    mean_psd_per_channel = []
    freqs = None

    for ch in range(num_channels):
        psds = []
        for i in range(batch_size):
            signal_ch = batch[i, 0, ch, :].cpu().numpy()
            f, pxx = welch(signal_ch, fs=fs, nperseg=nperseg)
            psds.append(pxx)
        psds = np.array(psds)
        mean_psd = psds.mean(axis=0)
        mean_psd_per_channel.append(mean_psd)
        if freqs is None:
            freqs = f

    return freqs, mean_psd_per_channel


def plot_mean_psd(
    freqs: np.ndarray,
    mean_psd_per_channel: List[np.ndarray],
    save_path: Optional[str] = None,
    show_legend: bool = True,
    max_channels_in_legend: int = 10
):
    """
    Plot the mean Power Spectral Density curves for all channels.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency values
    mean_psd_per_channel : list of np.ndarray
        Mean PSD for each channel
    save_path : str, optional
        Path to save figure
    show_legend : bool
        Whether to show legend
    max_channels_in_legend : int
        Maximum number of channels to show in legend
    """
    plt.figure(figsize=(12, 6))

    for ch, psd in enumerate(mean_psd_per_channel):
        label = f'Ch {ch+1}' if ch < max_channels_in_legend else None
        plt.semilogy(freqs, psd, alpha=0.6, label=label)

    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('PSD (V²/Hz)', fontsize=12)
    plt.title('Mean Power Spectral Density per Channel', fontsize=14, fontweight='bold')

    if show_legend and len(mean_psd_per_channel) <= max_channels_in_legend:
        plt.legend(loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_channel_statistics(
    batch: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plot mean and standard deviation for each channel.

    Parameters
    ----------
    batch : torch.Tensor
        Input batch of shape (batch_size, 1, num_channels, segment_length)
    save_path : str, optional
        Path to save figure
    """
    samples_squeezed = batch.squeeze(1)  # (batch_size, n_channels, n_times)

    channel_means = samples_squeezed.mean(dim=(0, 2)).cpu().numpy()
    channel_stds = samples_squeezed.std(dim=(0, 2)).cpu().numpy()

    channels = np.arange(1, samples_squeezed.shape[1] + 1)

    plt.figure(figsize=(12, 5))
    plt.errorbar(
        channels, channel_means, yerr=channel_stds,
        fmt='o', capsize=5, color='steelblue', ecolor='lightblue'
    )
    plt.title("Mean ± STD per Channel (Averaged over Batch & Time)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_parameter_comparison(
    base_trainable: int,
    base_total: int,
    lora_trainable: int,
    lora_total: int,
    save_path: Optional[str] = None
):
    """
    Plot parameter comparison between base and LoRA models.

    Parameters
    ----------
    base_trainable : int
        Number of trainable parameters in base model
    base_total : int
        Total parameters in base model
    lora_trainable : int
        Number of trainable parameters in LoRA model
    lora_total : int
        Total parameters in LoRA model
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Parameter comparison
    models = ['Base Model\n(Full Fine-tuning)', 'LoRA Model\n(Adapter Fine-tuning)']
    trainable_params = [base_trainable, lora_trainable]
    frozen_params = [base_total - base_trainable, lora_total - lora_trainable]

    x = range(len(models))
    width = 0.5

    ax1.bar(x, frozen_params, width, label='Frozen', color='lightgray')
    ax1.bar(x, trainable_params, width, bottom=frozen_params,
            label='Trainable', color='cornflowerblue')
    ax1.set_ylabel('Number of Parameters', fontsize=12)
    ax1.set_title('Parameter Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    for i, (train, total) in enumerate([(base_trainable, base_total),
                                         (lora_trainable, lora_total)]):
        ax1.text(i, total + max(trainable_params) * 0.02,
                 f'{100*train/total:.1f}%\ntrainable',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Training efficiency
    reduction = 100 * (1 - lora_trainable / base_trainable)
    bars = ax2.bar(['Full Fine-tuning', 'LoRA Fine-tuning'],
                   [100, 100 - reduction],
                   color=['coral', 'seagreen'], width=0.6)
    ax2.set_ylabel('Relative Training Cost (%)', fontsize=12)
    ax2.set_title('Training Efficiency Gain', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 120])
    ax2.grid(axis='y', alpha=0.3)

    ax2.text(0, 105, '100%', ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, 100 - reduction + 5, f'{100-reduction:.1f}%',
             ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = 'Accuracy',
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.

    Parameters
    ----------
    train_losses : list of float
        Training loss values
    val_losses : list of float
        Validation loss values
    train_metrics : list of float, optional
        Training metric values
    val_metrics : list of float, optional
        Validation metric values
    metric_name : str
        Name of the metric
    save_path : str, optional
        Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)

    if train_metrics is not None and val_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot metric if provided
    if train_metrics is not None and val_metrics is not None:
        ax2.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}', linewidth=2)
        ax2.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title(f'Training and Validation {metric_name}',
                      fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list of str, optional
        Names of classes
    save_path : str, optional
        Path to save figure
    normalize : bool
        Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def visualize_sample_signals(
    batch: torch.Tensor,
    num_samples: int = 3,
    num_channels: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize sample EEG signals.

    Parameters
    ----------
    batch : torch.Tensor
        Input batch of shape (batch_size, 1, num_channels, segment_length)
    num_samples : int
        Number of samples to visualize
    num_channels : int
        Number of channels to show per sample
    save_path : str, optional
        Path to save figure
    """
    batch_np = batch.cpu().numpy()
    num_samples = min(num_samples, batch.shape[0])
    num_channels = min(num_channels, batch.shape[2])

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        for ch in range(num_channels):
            signal = batch_np[i, 0, ch, :]
            ax.plot(signal, label=f'Ch {ch+1}', alpha=0.7)

        ax.set_xlabel('Time (samples)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
