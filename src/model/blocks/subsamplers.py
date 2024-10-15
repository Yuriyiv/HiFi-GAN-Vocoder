from torch import Tensor, nn


class BaseConvSubsampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        ker_size: int = 3,
        stride: int = 2,
        depthwise: bool = False,
    ) -> None:
        """
        Base class for ConvSubsampler with the option to use depthwise separable convolutions.
        Padded version for simpler pipeline.
        Reduce frame rate from hop_length to 1/4 * hop_length
        For example, with hop_length = 160 (10 ms) we get 40 ms after output.

        Args:
            input_dim (int): The number of input channels (usually 1 for spectrograms).
            out_dim (int): The number of output channels.
            ker_size (int): The size of the convolution kernel. Defaults to 3.
            stride (int): The stride of the convolution. Defaults to 2.
            depthwise (bool): Whether to use depthwise separable convolutions. Defaults to False.
        """
        super().__init__()

        if depthwise:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    out_dim,
                    kernel_size=ker_size,
                    stride=stride,
                    groups=input_dim,
                    padding="same",
                ),
                nn.SiLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, padding="same"),
                nn.SiLU(),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    out_dim,
                    kernel_size=ker_size,
                    stride=stride,
                    padding="same",
                ),
                nn.SiLU(),
                nn.Conv2d(
                    out_dim,
                    out_dim,
                    kernel_size=ker_size,
                    stride=stride,
                    padding="same",
                ),
                nn.SiLU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor (#batch, time, freq).

        Returns:
            Tensor: Subsampled and encoded tensor (#batch, new_time, new_dim).
        """
        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # [batch_size, 1, time, freq]

        x = self.conv_layers(x)  # [batch_size, channel_dim, time, freq]

        return x


class ConvSubsampler(nn.Module):
    """
    Basic Convolution Subsampler.
    Padded version for simpler pipeline.
    Reduce frame rate from hop_length to 1/4 * hop_length
    For example, with hop_length = 160 (10 ms) we get 40 ms after output.

    Args:
        input_dim (int): The number of input channels (usually 1 for spectrograms).
        out_dim (int): The number of output channels.
        ker_size (int): The size of the convolution kernel. Defaults to 3.
        stride (int): The stride of the convolution. Defaults to 2.
        depthwise (bool): Whether to use depthwise separable convolutions. Defaults to False.
    """

    def __init__(
        self, input_dim: int, out_dim: int, ker_size: int = 3, stride: int = 2
    ) -> None:
        """
        With default params subsamples with ~1/4x rate.
        As o_1 = floor[(i - k) / s] + 1 = floor[(i - 3) / 2] + 1
        And o_2 = floor[(i - k) / s] + 1 = floor[(i - 3) / 2] + 1
        """
        super().__init__(input_dim, out_dim, ker_size, stride, depthwise=False)


class ConvDepthwiseSubsampler(nn.Module):
    """
    Depthwise Convolution Subsampler.
    Padded version for simpler pipeline.
    Reduce frame rate from hop_length to 1/4 * hop_length
    For example, with hop_length = 160 (10 ms) we get 40 ms after output.

    Args:
        input_dim (int): The number of input channels (usually 1 for spectrograms).
        out_dim (int): The number of output channels.
        ker_size (int): The size of the convolution kernel. Defaults to 3.
        stride (int): The stride of the convolution. Defaults to 2.
        depthwise (bool): Whether to use depthwise separable convolutions. Defaults to False.
    """

    def __init__(
        self, input_dim: int, out_dim: int, ker_size: int = 3, stride: int = 2
    ) -> None:
        """
        With default params subsamples with ~1/4x rate.
        Reduce numbers of params due to using Depthwise separation tecqnique
        """
        super().__init__(input_dim, out_dim, ker_size, stride, depthwise=True)
