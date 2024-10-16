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
        Not padded version for simpler pipeline.
        Reduce frame rate from hop_length to 1/4 * hop_length
        For example, with hop_length = 160 (10 ms) we get 40 ms after output.

        Args:
            out_dim (int): The number of output channels.
            ker_size (int): The size of the convolution kernel. Defaults to 3.
            stride (int): The stride of the convolution. Defaults to 2.
            depthwise (bool): Whether to use depthwise separable convolutions. Defaults to False.
        """
        super().__init__()
        # self.pad = nn.ConstantPad2d((0, 0, 0, 4), 0.0)
        self.ker_size = ker_size
        self.stride = stride
        self.depthwise = depthwise
        # output_dim = compute_output_shape(input_dim, ker_size, stride)
        # self.output_dim = compute_output_shape(output_dim, ker_size, stride)

        if depthwise:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    1, out_dim, kernel_size=ker_size, stride=stride, groups=1, padding=0
                ),
                nn.SiLU(),
                nn.Conv2d(
                    out_dim, out_dim, kernel_size=1, stride=stride
                ),  # , padding="same"
                nn.SiLU(),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, out_dim, kernel_size=ker_size, stride=stride),
                nn.SiLU(),
                nn.Conv2d(out_dim, out_dim, kernel_size=ker_size, stride=stride),
                nn.SiLU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor (#batch, time, freq).

        Returns:
            Tensor: Subsampled and encoded tensor (#batch, channel_dim, new_time, new_dim).
        """
        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # [batch_size, 1, time, freq]

        x = self.conv_layers(
            x
        )  # [batch_size, channel_dim (=out_dim), time_subsampled, freq_subsampled]

        return x

    def get_info(self):
        return {
            "stride": self.stride,
            "ker_size": self.ker_size,
            "depthwise": self.depthwise,
        }


class ConvSubsampler(BaseConvSubsampler):
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

    def get_stride(self):
        return super().get_info()["stride"]

    def get_ker_size(self):
        return super().get_info()["ker_size"]

    def get_depthwise_status(self):
        return super().get_info()["depthwise"]


class ConvDepthwiseSubsampler(BaseConvSubsampler):
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

    def get_stride(self):
        return super().get_info()["stride"]

    def get_ker_size(self):
        return super().get_info()["ker_size"]

    def get_depthwise_status(self):
        return super().get_info()["depthwise"]
