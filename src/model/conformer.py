from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchtune.modules import RotaryPositionalEmbeddings

from src.model import BaseModelABC
from src.model.blocks import ConvDepthwiseSubsampler  # ConvSubsampler
from src.utils import compute_output_shape


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

    def forward(self, x: Tensor) -> Tensor:
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

        def _attention_compute(q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0)

        possible_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

        with sdpa_kernel(possible_backends):
            attention = _attention_compute(
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

    def forward(self, x: Tensor) -> Tensor:
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
    def __init__(
        self, input_dim: int, output_dim: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the multi-head self-attention module with layer normalization and dropout.

        Args:
            input_dim (int): The dimensionality of the input.
            output_dim (int): The dimensionality of the output.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate to be applied after the attention layer.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.multi_head_self_attention = MultiHeadAttention(
            input_dim, output_dim, num_heads
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the multi-head self-attention module.

        Args:
            x (Tensor): Input tensor of shape (N, L, input_dim)

        Returns:
            Tensor: Output tensor of shape (N, L, output_dim) with added residual connection.
        """
        resid = x
        x = self.layer_norm(x)
        x = self.multi_head_self_attention(x)
        x = self.dropout(x)
        return resid + x


class ConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        conv_expansion_factor: int,
        depthwise_ker_size: int,
        bias: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the convolution module with normalization, GatedLinearUnit activation, and depthwise convolutions.

        Args:
            dim (int): Input/output dimensionality.
            conv_expansion_factor (int): Expansion factor for pointwise convolutions.
            depthwise_ker_size (int): Kernel size for the depthwise convolution.
            bias (bool): Whether to use bias in convolution layers. Defaults to False.
        """
        super().__init__()
        assert (
            conv_expansion_factor % 2 == 0
        ), "conv_expansion_factor must be divisible by 2"
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_first = nn.Conv1d(
            dim, dim * conv_expansion_factor, kernel_size=1, bias=bias
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            dim * conv_expansion_factor // 2,
            dim * conv_expansion_factor // 2,
            kernel_size=depthwise_ker_size,
            padding="same",
            bias=bias,
            groups=dim * conv_expansion_factor // 2,
        )
        self.batch_norm = nn.BatchNorm1d(dim * conv_expansion_factor // 2)
        self.swish = nn.SiLU()
        self.pointwise_second = nn.Conv1d(
            dim * conv_expansion_factor // 2, dim, kernel_size=1, bias=bias
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the convolution module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, channels).

        Returns:
            Tensor: Output tensor with the same shape as input, after convolution and residual connection.
        """
        resid = x
        x = self.layer_norm(x)
        x = x.permute(
            (0, 2, 1)
        )  # (batch_size, length, channels) -> (batch_size, channels, length)
        x = self.pointwise_first(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_second(x)
        x = self.dropout(x)
        return x.permute((0, 2, 1)) + resid


class FeedForwardModule(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, drop_rate: float = 0.1, bias: bool = True
    ) -> None:
        """
        Initializes the feed-forward module with layer normalization and SiLU activation.

        Args:
            input_dim (int): Dimensionality of the input.
            hidden_dim (int): Dimensionality of the hidden layer.
            drop_rate (float): Dropout rate. Defaults to 0.1.
            bias (bool): Whether to use bias in linear layers. Defaults to True.
        """
        super().__init__()
        assert isinstance(
            input_dim, int
        ), f"input_dim should be int, got {type(input_dim)}"
        assert isinstance(
            hidden_dim, int
        ), f"hidden_dim should be int, got {type(hidden_dim)}"
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, input_dim, bias=bias),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the FeedForward module.

        Args:
            x (Tensor): Input tensor of shape (*, input_dim).

        Returns:
            Tensor: Output tensor after feed-forward of shape (*, input_dim).
        """
        return self.ff_layer(x)


class ConformerBlock(nn.Module):
    """
    A single block in the modified Conformer architecture that includes feed-forward layers,
    multi-head self-attention, and convolutional modules.
    Originally proposed in https://arxiv.org/abs/2005.08100

    It applies the following structure:
    1. Feed-forward module
    2. Multi-head self-attention
    3. Convolutional module
    4. Feed-forward module
    5. Layer normalization

    The block capturing both local and global dependencies using
    a combination of attention and convolution mechanisms.

    Args:
        input_dim (int): The dimensionality of the input and output features.
        feed_forw_dim (int): The dimensionality of the hidden layer in the feed-forward modules.
        num_att_heads (int): The number of attention heads in the multi-head self-attention module.
        conv_expansion_factor (int): The expansion factor for the convolutional layers.
        depthwise_ker_size (int): The kernel size for depthwise convolutions in the ConvModule.
    """

    def __init__(
        self,
        input_dim: int,
        feed_forw_dim: int,
        num_att_heads: int,
        conv_expansion_factor: int,
        depthwise_ker_size: int,
    ) -> None:
        """
        Initializes the components of the ConformerBlock
        """
        super().__init__()

        assert isinstance(
            input_dim, int
        ), f"input_dim should be int, got {type(input_dim)}"
        assert isinstance(
            feed_forw_dim, int
        ), f"feed_forw_dim should be int, got {type(feed_forw_dim)}"
        assert isinstance(
            num_att_heads, int
        ), f"num_att_heads should be int, got {type(num_att_heads)}"
        assert isinstance(
            conv_expansion_factor, int
        ), f"conv_expansion_factor should be int, got {type(conv_expansion_factor)}"
        assert isinstance(
            depthwise_ker_size, int
        ), f"depthwise_ker_size should be int, got {type(depthwise_ker_size)}"

        self.feed_forward_first = FeedForwardModule(
            input_dim, feed_forw_dim
        )  # input -> input
        self.multi_head_self_attention = MultiHeadSelfAttentionModule(
            input_dim, input_dim, num_att_heads
        )
        self.conv_block = ConvModule(
            input_dim, conv_expansion_factor, depthwise_ker_size
        )
        self.feed_forward_second = FeedForwardModule(input_dim, feed_forw_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the ConformerBlock.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time (seq_length), input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, time (seq_length), input_dim).
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
    """
    The ConformerEncoder is a stack of ConformerBlocks, each block consisting of:
    - Feed-forward module
    - Multi-head self-attention
    - Convolutional module
    - Layer normalization

    The encoder processes input sequences by passing them through a series of Conformer blocks
    after subsampling and linear layer.

    Args:
        input_dim (int): The dimensionality of the input features (embedding dimension).
        num_conform_blocks (int): The number of Conformer blocks to stack in the encoder.
        num_att_heads (int): The number of attention heads in the multi-head self-attention module.
        depthwise_ker_size (int): The kernel size for the depthwise convolution in the convolutional module.
        conv_expansion_factor (int): Expansion factor for the pointwise convolution in the convolutional module.
        feed_forw_dim (Optional[int]): The hidden dimensionality of the feed-forward module. Defaults to 4 * input_dim if not specified.
        pos_enc (nn.Module): Positional encoding module. By default use RotaryPositionalEmbeddings from torchtune
    """

    def __init__(
        self,
        input_dim: int,
        num_conform_blocks: int,
        num_att_heads: int,
        depthwise_ker_size: int,
        conv_expansion_factor: int = 2,
        feed_forw_dim: Optional[int] = None,
        pos_encoder: nn.Module = RotaryPositionalEmbeddings,
        rotary_dim: Optional[int] = None,
    ) -> None:
        """
        Initializes the ConformerEncoder
        """
        super().__init__()
        # self.rotary_dim = rotary_dim or feed_forw_dim
        feed_forw_dim = feed_forw_dim or 4 * input_dim
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    input_dim,
                    feed_forw_dim,
                    num_att_heads,
                    conv_expansion_factor,
                    depthwise_ker_size,
                )
                for _ in range(num_conform_blocks)
            ]
        )
        # self.pos_enc = pos_encoder(rotary_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the ConformerEncoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time (seq_length), input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, time (seq_length), input_dim).
        """
        # x = self.pos_enc(x)
        for block in self.conformer_blocks:
            x = block(x)
        return x


class ConformerModel(BaseModelABC):
    """
    A Conformer implementation. Originally proposed in https://arxiv.org/abs/2005.08100
    Conformer is a neural network architecture that combines Conformer encoders with
    a decoder to perform tasks such as speech recognition or sequence-to-sequence modeling.

    The model processes input spectrograms through a subsampler, a stack of Conformer blocks,
    and a decoder to produce output logits over a vocabulary of tokens.

    Args:
        encoder_dim (int): The dimensionality of embedding features.
        decoder_dim (int): The dimensionality of the decoder's hidden states.
        n_tokens (int): Number of tokens in the vocabulary.
        num_conform_blocks (int): The number of Conformer blocks to stack in the encoder.
        num_att_heads (int): The number of attention heads in the multi-head self-attention module.
        ker_size (int): The kernel size for the depthwise convolution in the convolutional module.
        conv_expansion_factor (int): Expansion factor for the pointwise convolution in the convolutional module.
        feed_forw_dim (Optional[int]): The hidden dimensionality of the feed-forward module. Defaults to 4 * input_dim if not specified.
        pos_enc (nn.Module): Positional encoding module. By default use RotaryPositionalEmbeddings from torchtune.
        subsampler_cls (nn.Module): Subsampling module class. Defaults to ConvDepthwiseSubsampler.
        decoder_cls (nn.Module): Decoder module class. Defaults to nn.LSTM.

    Attributes:
        subsampler (nn.Module): Subsampling layer to reduce the temporal resolution of the input spectrogram.
        linear_projection (nn.Linear): Linear layer to project the subsampled features to the encoder dimension.
        dropout (nn.Dropout): Dropout layer for regularization after projection.
        encoder (ConformerEncoder): Stack of Conformer blocks for encoding the input sequence.
        decoder (nn.Module): Decoder module (e.g., LSTM) for processing encoded features.
        classification_head (nn.Linear): Linear layer to map decoder outputs to vocabulary logits.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_conform_blocks: int,
        num_att_heads: int,
        ker_size: int,
        n_tokens: int,
        freq: int,
        conv_expansion_factor: int = 2,
        feed_forw_dim: Optional[int] = None,
        pos_encoder: nn.Module = RotaryPositionalEmbeddings,
        subsampler_cls: nn.Module = ConvDepthwiseSubsampler,
        decoder_cls: nn.Module = nn.LSTM,
    ) -> None:
        """
        Args:
            input_dim (int): the dimensionality of the input features.

            fc_hidden (int): number of hidden features.
        """
        super().__init__()
        feed_forw_dim = feed_forw_dim or 4 * encoder_dim
        self.encoder_dim = encoder_dim
        self.freq = freq

        self.subsampler = subsampler_cls(input_dim=freq, out_dim=encoder_dim)
        new_freq = self.get_subsampler_output_dim(self.freq, self.subsampler)
        self.linear_projection = nn.Linear(encoder_dim * new_freq, encoder_dim)
        self.dropout = nn.Dropout(0.1)
        self.encoder = ConformerEncoder(
            encoder_dim,
            num_conform_blocks,
            num_att_heads,
            ker_size,
            conv_expansion_factor,
            feed_forw_dim,
            pos_encoder,
        )
        self.decoder = decoder_cls(encoder_dim, decoder_dim, batch_first=True)
        self.classification_head = nn.Linear(
            in_features=decoder_dim, out_features=n_tokens
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram. The shape: [batch_size, n_mels (freq), time]
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            Dict[str, Tensor]: output dict containing log_probs and
                transformed lengths.
        """
        x = spectrogram.transpose(1, 2)  # [batch_size, time, n_mels (freq)]
        x = self.subsampler(
            x
        )  # [batch_size, encoder_dim, time_subsampled, freq_subsampled]
        x = x.transpose(1, 2).flatten(
            2
        )  # [batch_size, time_subsampled, encoder_dim * freq_subsampled]
        x = self.linear_projection(x)  # [batch_size, time_subsampled, encoder_dim]
        x = self.dropout(x)
        x_encoded = self.encoder(x)  # [batch_size, time_subsampled, encoder_dim]
        x_decoded, _ = self.decoder(
            x_encoded
        )  # [batch_size, time_subsampled, decoder_dim]
        output = self.classification_head(
            x_decoded
        )  # [batch_size, time_subsampled, n_tokens]
        log_probs = nn.functional.log_softmax(
            output, dim=-1
        )  # [batch_size, time_subsampled, n_tokens]
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        # print(input_lengths)
        return self.get_subsampler_output_dim(input_lengths, self.subsampler)

    def get_subsampler_output_dim(self, input_dim: Tensor, sampler) -> Tensor:
        """
        Computes the output dimension after applying the subsampler.

        Args:
            input_dim (Tensor): Input lengths. Shape: [batch_size]
            sampler (nn.Module): The subsampling module.

        Returns:
            Tensor: Output lengths after subsampling. Shape: [batch_size]
        """
        depthwise = sampler.get_depthwise_status()
        ker_size_2 = 1 if depthwise else sampler.get_ker_size()
        output_dim = compute_output_shape(
            input_dim, sampler.get_ker_size(), sampler.get_stride()
        )
        output_dim = compute_output_shape(output_dim, ker_size_2, sampler.get_stride())
        return output_dim
