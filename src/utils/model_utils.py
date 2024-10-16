import math
from typing import Union

import torch


def compute_output_shape(
    input_shape: Union[int, torch.Tensor],
    ker_size: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
) -> Union[int, torch.Tensor]:
    """
    Computes the output shape of a convolutional layer for a given input shape.

    This function supports two types of inputs:
    1. Integer (int): Represents a single dimension (e.g., height or width).
       - The function computes the output dimension and returns it as an integer.
    2. Tensor (torch.Tensor): Represents multiple dimensions.
       - The function computes the output dimension for each element in the tensor and returns a tensor of output dimensions.

    Args:
        input_shape (int or torch.Tensor): The input shape. Can be an integer or a tensor of integers.
        ker_size (int): The size of the convolutional kernel (filter).
        stride (int): The stride of the convolution.
        padding (int, optional): The amount of padding applied to the input. Defaults to 0.
        dilation (int, optional): Dilation factor for the convolution kernel. Defaults to 1.

    Returns:
        int or torch.Tensor: The computed output dimension(s).
            - If input_shape is an int, returns an int.
            - If input_shape is a torch.Tensor, returns a tensor of output dimensions.
    """
    if isinstance(input_shape, int):
        output = math.floor(
            (input_shape + 2 * padding - dilation * (ker_size - 1) - 1) / stride + 1
        )
        return output

    elif isinstance(input_shape, torch.Tensor):
        if not torch.is_floating_point(input_shape):
            input_shape = input_shape.float()

        output = torch.floor(
            (input_shape + 2 * padding - dilation * (ker_size - 1) - 1) / stride + 1
        )

        output = output.to(torch.int)

        return output

    else:
        raise TypeError("input_shape must be an int or torch.Tensor")
