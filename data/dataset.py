"""Dataset classes for ECoG/EEG data loading."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Callable


class MultiSubjectECoGDataset(Dataset):
    """PyTorch Dataset for loading multi-subject ECoG/EEG data from .fif files."""

    def __init__(
        self,
        file_list: List[str],
        freq_band: Optional[Tuple[float, float]] = (1, 100),
        remove_bad: bool = True,
        scale: bool = True,
        transform: Optional[Callable] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        file_list : list of str
            List of .fif file paths.
        freq_band : tuple, optional
            Bandpass filter frequency range in Hz, e.g., (1, 100).
            Set to None to skip filtering.
        remove_bad : bool, optional
            Remove bad channels marked in the data.
        scale : bool, optional
            Apply StandardScaler per channel.
        transform : callable, optional
            Optional transformation to apply to each sample.
        time_range : tuple or None, optional
            Time window (start_sec, end_sec) relative to event onset to extract.
            Example: (0.5, 3). If None, defaults to (0, 4).
        """
        self.transform = transform
        self.data = []
        self.labels = []
        self.scale = scale
        self.remove_bad = remove_bad
        self.freq_band = freq_band
        self.time_range = time_range

        for file in file_list:
            if os.path.exists(file):
                print(f'Loading and processing {file}')
                self._load_subject_data(file)
            else:
                print(f'Warning: File not found: {file}')

        if len(self.data) == 0:
            raise ValueError("No data loaded. Please check file paths.")

        self.data = np.array(self.data)  # (n_trials, n_channels, n_times)
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.data)} trials from {len(file_list)} file(s).")
        print(f"Data shape: {self.data.shape}")
        print(f"Unique labels: {np.unique(self.labels)}")

    def _load_subject_data(self, file_path: str):
        """Load and process a single subject's data file."""
        # Load raw data
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        if self.remove_bad:
            raw.pick_types(ecog=True, eeg=True, exclude='bads')

        # Apply bandpass filter
        if self.freq_band:
            raw.filter(
                self.freq_band[0],
                self.freq_band[1],
                fir_design='firwin',
                verbose=False
            )

        # Get events and labels
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        print(f"  Found {len(events)} events, event_id mapping: {event_id}")

        # Filter out REST events (optional - keep only MI events)
        mi_event_ids = {}
        for name, code in event_id.items():
            # Uncomment the line below to filter out rest events
            # if 'rest' not in name.lower():
            mi_event_ids[name] = code

        if not mi_event_ids:
            print("  No Motor Imagery events found in this file.")
            return

        print(f"  Using events: {list(mi_event_ids.keys())}")

        # Epoching
        if self.time_range:
            tmin, tmax = self.time_range
        else:
            tmin, tmax = 0, 4  # Default 4s trial length

        epochs = mne.Epochs(
            raw, events, event_id=mi_event_ids,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False
        )

        X = epochs.get_data()  # (n_trials, n_channels, n_times)
        y = epochs.events[:, -1]  # Label indices

        # Optional scaling per channel
        if self.scale:
            for ch in range(X.shape[1]):
                scaler = StandardScaler()
                X[:, ch, :] = scaler.fit_transform(X[:, ch, :])

        # Save data and labels
        for trial, label in zip(X, y):
            self.data.append(trial)
            self.labels.append(label - 1)  # Zero-based labels

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Parameters
        ----------
        index : int
            Sample index

        Returns
        -------
        sample : torch.Tensor
            EEG data of shape (1, n_channels, n_times)
        label : torch.Tensor
            Class label
        """
        sample = self.data[index]  # (n_channels, n_times)
        label = self.labels[index]

        # Convert to PyTorch tensor and add channel dimension (1, n_channels, n_times)
        sample = torch.from_numpy(sample).float().unsqueeze(0)
        label = torch.tensor(label).long()

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    @property
    def num_channels(self) -> int:
        """Return number of EEG channels."""
        return self.data.shape[1]

    @property
    def num_timepoints(self) -> int:
        """Return number of time points per trial."""
        return self.data.shape[2]

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(np.unique(self.labels))


def create_file_list(
    data_prefix: str,
    subject_ids: List[int],
    session_ids: List[int],
    pattern: str = "Subject {subject}/{session}/{session}-raw.fif"
) -> List[str]:
    """
    Create a list of file paths for given subjects and sessions.

    Parameters
    ----------
    data_prefix : str
        Root directory containing the data
    subject_ids : list of int
        List of subject IDs
    session_ids : list of int
        List of session IDs
    pattern : str
        File path pattern with {subject} and {session} placeholders

    Returns
    -------
    file_list : list of str
        List of file paths
    """
    file_list = []
    for subject in subject_ids:
        for session in session_ids:
            file_path = os.path.join(
                data_prefix,
                pattern.format(subject=subject, session=session)
            )
            file_list.append(file_path)
    return file_list
