import torch
from torch import Tensor, nn


class LogTransform(nn.Module):
    """
    Torch.log wrapper.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Initializes the LogTransform.

        Args:
            eps (float): A small constant to add to the input to prevent taking the log of zero.
                         Defaults to 1e-10.
        """
        super().__init__()
        self.eps = eps

    def __call__(self, mel_spectrogram: Tensor) -> Tensor:
        """
        Applies a logarithmic transformation to the mel-spectrogram.

        Args:
            mel_spectrogram (Tensor): Input tensor of shape (batch_size, freq_bins, max_time_steps).

        Returns:
            Tensor: Log-transformed tensor with the same shape as the input.
        """
        if torch.any(mel_spectrogram < 0):
            raise ValueError("Input mel_spectrogram contains negative values")
        return torch.log(mel_spectrogram + self.eps)
