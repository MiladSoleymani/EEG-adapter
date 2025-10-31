"""ATCNet model implementation for EEG/ECoG classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATCNet(nn.Module):
    """
    Attention-based Temporal Convolutional Network for EEG classification.

    Combines convolutional feature extraction with multi-head self-attention
    and temporal convolutional networks for robust EEG signal classification.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        num_windows: int = 3,
        num_electrodes: int = 22,
        conv_pool_size: int = 7,
        F1: int = 16,
        D: int = 2,
        tcn_kernel_size: int = 4,
        tcn_depth: int = 2,
        chunk_size: int = 1125
    ):
        """
        Initialize ATCNet.

        Parameters
        ----------
        in_channels : int
            Number of input channels (typically 1 for EEG)
        num_classes : int
            Number of output classes
        num_windows : int
            Number of sliding windows for attention
        num_electrodes : int
            Number of EEG electrodes/channels
        conv_pool_size : int
            Pooling size for convolutional block
        F1 : int
            Number of temporal filters
        D : int
            Depth multiplier for depthwise convolution
        tcn_kernel_size : int
            Kernel size for temporal convolutional layers
        tcn_depth : int
            Number of TCN blocks per window
        chunk_size : int
            Length of input time series
        """
        super(ATCNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_windows = num_windows
        self.num_electrodes = num_electrodes
        self.pool_size = conv_pool_size
        self.F1 = F1
        self.D = D
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_depth = tcn_depth
        self.chunk_size = chunk_size
        F2 = F1 * D

        # Convolutional feature extraction block
        self.conv_block = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(in_channels, F1, (1, int(chunk_size / 2 + 1)),
                      stride=1, padding='same', bias=False),
            nn.BatchNorm2d(F1, False),

            # Spatial (depthwise) convolution
            nn.Conv2d(F1, F2, (num_electrodes, 1), padding=0, groups=F1),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(0.1),

            # Separable convolution
            nn.Conv2d(F2, F2, (1, 16), bias=False, padding='same'),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size)),
            nn.Dropout2d(0.1)
        )

        self.__build_model()

    def __build_model(self):
        """Build attention and TCN modules dynamically."""
        with torch.no_grad():
            # Get dimensions after conv block
            x = torch.zeros(2, self.in_channels,
                            self.num_electrodes, self.chunk_size)
            x = self.conv_block(x)
            x = x[:, :, -1, :]
            x = x.permute(0, 2, 1)
            self.__chan_dim, self.__embed_dim = x.shape[1:]
            self.win_len = self.__chan_dim - self.num_windows + 1

            # Build modules for each sliding window
            for i in range(self.num_windows):
                st = i
                end = x.shape[1] - self.num_windows + i + 1
                x2 = x[:, st:end, :]

                # Multi-head self-attention
                self.__add_msa(i)
                x2_ = self.get_submodule("msa" + str(i))(x2, x2, x2)[0]
                self.__add_msa_drop(i)
                x2_ = self.get_submodule("msa_drop" + str(i))(x2)
                x2 = torch.add(x2, x2_)

                # Temporal convolutional blocks
                for j in range(self.tcn_depth):
                    self.__add_tcn((i + 1) * j, x2.shape[1])
                    out = self.get_submodule("tcn" + str((i + 1) * j))(x2)
                    if x2.shape[1] != out.shape[1]:
                        self.__add_recov(i)
                        x2 = self.get_submodule("re" + str(i))(x2)
                    x2 = torch.add(x2, out)
                    x2 = nn.ELU()(x2)

                x2 = x2[:, -1, :]
                self.__dense_dim = x2.shape[-1]

                # Classification head for this window
                self.__add_dense(i)
                x2 = self.get_submodule("dense" + str(i))(x2)

    def __add_msa(self, index: int):
        """Add multi-head self-attention module."""
        self.add_module(
            'msa' + str(index),
            nn.MultiheadAttention(
                embed_dim=self.__embed_dim,
                num_heads=2,
                batch_first=True
            )
        )

    def __add_msa_drop(self, index: int):
        """Add dropout after attention."""
        self.add_module('msa_drop' + str(index), nn.Dropout(0.3))

    def __add_tcn(self, index: int, num_electrodes: int):
        """Add temporal convolutional network block."""
        self.add_module(
            'tcn' + str(index),
            nn.Sequential(
                nn.Conv1d(num_electrodes, 32,
                          self.tcn_kernel_size, padding='same'),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Conv1d(32, 32, self.tcn_kernel_size, padding='same'),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3)
            )
        )

    def __add_recov(self, index: int):
        """Add recovery convolution for dimension matching."""
        self.add_module(
            're' + str(index),
            nn.Conv1d(self.win_len, 32, 4, padding='same')
        )

    def __add_dense(self, index: int):
        """Add classification head."""
        self.add_module(
            'dense' + str(index),
            nn.Linear(self.__dense_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, num_electrodes, chunk_size)

        Returns
        -------
        torch.Tensor
            Class logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        x = self.conv_block(x)
        x = x[:, :, -1, :]
        x = x.permute(0, 2, 1)

        # Process each sliding window
        for i in range(self.num_windows):
            st = i
            end = x.shape[1] - self.num_windows + i + 1
            x2 = x[:, st:end, :]

            # Multi-head self-attention with residual
            x2_ = self.get_submodule("msa" + str(i))(x2, x2, x2)[0]
            x2_ = self.get_submodule("msa_drop" + str(i))(x2)
            x2 = torch.add(x2, x2_)

            # Temporal convolutional blocks with residuals
            for j in range(self.tcn_depth):
                out = self.get_submodule("tcn" + str((i + 1) * j))(x2)
                if x2.shape[1] != out.shape[1]:
                    x2 = self.get_submodule("re" + str(i))(x2)
                x2 = torch.add(x2, out)
                x2 = nn.ELU()(x2)

            # Get final time step and classify
            x2 = x2[:, -1, :]
            x2 = self.get_submodule("dense" + str(i))(x2)

            # Ensemble predictions from all windows
            if i == 0:
                sw_concat = x2
            else:
                sw_concat = sw_concat.add(x2)

        # Average predictions from all windows
        x = sw_concat / self.num_windows
        return x
