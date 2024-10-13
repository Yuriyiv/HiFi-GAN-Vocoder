from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.model import BaselineModel


class Attention(nn.Module):
    """
    The class is a wrapper over an efficient implementation of the dot product attention mechanism on the corresponding CUDA kernels.
    It tries using different ways to compute attetion in following order sorting by efficiency:
        1. FlashAttention 2.0 implementation
        2. The memory efficient implementation from xformers
        3. PyTorch default attention implementation

    Attributes:
        query (nn.Linear): Linear layer to project input to query vectors.
        key (nn.Linear): Linear layer to project input to key vectors.
        value (nn.Linear): Linear layer to project input to value vectors.
        mask (Optional[Tensor]): Optional mask tensor to apply to the attention scores.
    """

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        value_dim: int,
        mask: Optional[Tensor] = None,
    ) -> None:
        """
        Initialize linear projection layers. The formulas:
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        Args:
            input_dim (int): The dimensionality of the input features.
            query_dim (int): The dimensionality of the query and key vectors.
            value_dim (int): The dimensionality of the value vectors.
            mask (Optional[Tensor], optional): A mask tensor to apply to the attention scores. Defaults to None.
        """
        super().__init__()
        self.query = nn.Linear(input_dim, query_dim, bias=False)
        self.key = nn.Linear(input_dim, query_dim, bias=False)
        self.value = nn.Linear(input_dim, value_dim, bias=False)
        self.mask = mask

    def forward(self, x) -> Tensor:
        """
        Performs the forward pass of the Attention module.
        Args:
            x (Tensor): Input tensor of shape (N, ..., L, input_dim), where:
                - N: Batch size
                - ...: Any number of additional batch dimensions
                - L: Target sequence length
                - input_dim: Dimensionality of input features

        Returns:
            Tensor: Output tensor after applying attention, of shape (N, ..., L, value_dim).
            In this implementation input_dim = value_dim
        """

        def attention_compute(q, k, v, mask):
            F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0)

        possible_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

        with sdpa_kernel(possible_backends):
            attention = attention_compute(
                self.query(x), self.key(x), self.value(x), self.mask
            )

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int) -> None:
        """
        Initialize MultiHeadAttention with input_dim, output_dim and num_heads params.

        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The desired dimensionality of the output features.
            num_heads (int): The number of attention heads.
        """
        super().__init__()

        self.head_dim = output_dim // num_heads
        assert (
            num_heads * self.head_dim == output_dim
        ), "output_dim have to be divisible by num_heads"

        self.heads = nn.ModuleList(
            [
                Attention(input_dim, self.head_dim, self.head_dim)
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x) -> Tensor:
        """
        Performs the forward pass of the MultiHeadAttention module.

        Args:
            x (Tensor): Input tensor of shape (N, L, input_dim), where:
                - N: Batch size
                - L: Sequence length
                - input_dim: Dimensionality of input features

        Returns:
            Tensor: Output tensor after applying multi-head attention, of shape (N, L, output_dim).
        """
        attention_list = [head(x) for head in self.heads]  # (N, L, heads_dim)

        concatted_heads = torch.cat(
            attention_list, dim=-1
        )  # (N, L, heads_dim * num_heads)

        return self.linear(concatted_heads)  # (N, L, output_dim)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.multi_head_self_attention = MultiHeadAttention(
            input_dim, output_dim, num_heads
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> Tensor:
        resid = x
        x = self.layer_norm(x)
        x = self.multi_head_self_attention(x)
        x = self.dropout(x)
        return resid + x


class ConvModule(nn.Module):
    def __init__(self, dim, conv_exp_factor, depthwise_ker_size, bias=False) -> None:
        super().__init__()
        assert conv_exp_factor % 2 == 0, "conv_exp_factor must be divisible by 2"
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_first = nn.Conv1d(
            dim, dim * conv_exp_factor, kernel_size=1, bias=bias
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            dim * conv_exp_factor // 2,
            dim * conv_exp_factor // 2,
            kernel_size=depthwise_ker_size,
            padding="same",
            bias=bias,
            groups=dim * conv_exp_factor // 2,
        )
        self.batch_norm = nn.BatchNorm1d(dim * conv_exp_factor // 2)
        self.swish = nn.SiLU()
        self.pointwise_second = nn.Conv1d(
            dim * conv_exp_factor // 2, dim, kernel_size=1, bias=bias
        )
        self.dropout = nn.Dropout()

    def forward(self, x) -> Tensor:
        resid = x
        x = self.layer_norm(x)
        # x = x.transpose(1, 2)
        x = self.pointwise_first(x)
        x = self.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_second(x)
        x = self.dropout(x)
        return x + resid


class FeedForwardModule(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, drop_rate: float = 0.1, bias: bool = True
    ) -> None:
        super().__init__()
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, input_dim, bias=bias),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        """
        Performs the forward pass of the FeedForward module.

        Args:
            x (Tensor): Input tensor of shape (*, input_dim).

        Returns:
            Tensor: Output tensor after feed-forward of shape (*, input_dim).
        """
        return self.ff_layer(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        feed_forw_dim,
        num_att_heads,
        conv_exp_factor,
        depthwise_ker_size,
    ) -> None:
        super().__init__()
        self.feed_forward_first = FeedForwardModule(
            input_dim, feed_forw_dim
        )  # input -> input
        self.multi_head_self_attention = MultiHeadSelfAttentionModule(
            input_dim, input_dim, num_att_heads
        )
        self.conv_block = ConvModule(input_dim, conv_exp_factor, depthwise_ker_size)
        self.feed_forward_second = FeedForwardModule(input_dim, feed_forw_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Performs the forward pass of the ConformerBlock.

        Args:
            x (Tensor): Input tensor of shape ().

        Returns:
            Tensor: Output tensor of  shape ().
        """
        resid = x
        x = self.feed_forward_first(x)
        x = resid + 0.5 * x
        x = self.multi_head_self_attention(x)  # residual inside
        x = self.conv_block(x)  # residual inside
        resid = x
        x = self.feed_forward_second(x)
        x = resid + 0.5 * x
        return self.layer_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_conform_blocks: int,
        num_att_heads: int,
        depthwise_ker_size: int,
        conv_exp_factor: int = 2,
        feed_forw_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        feed_forw_dim = feed_forw_dim or 4 * input_dim
        self.subsampler = nn.Identity()
        self.linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout()
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    input_dim,
                    feed_forw_dim,
                    num_att_heads,
                    conv_exp_factor,
                    depthwise_ker_size,
                )
                for _ in range(num_conform_blocks)
            ]
        )

    def forward(self, x) -> Tensor:
        """
        Performs the forward pass of the ConformerEncoder.

        Args:
            x (Tensor): Input tensor of shape ().

        Returns:
            Tensor: Output tensor of  shape ().
        """
        x = self.linear(x)
