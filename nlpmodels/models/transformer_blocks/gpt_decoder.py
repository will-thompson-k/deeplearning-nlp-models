"""
This module contains the Decoder layer and composite decoder stack (made of many decoder layers) for the
GPT-specific Transformer model architecture.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from nlpmodels.models.transformer_blocks.attention import MultiHeadedAttention
from nlpmodels.models.transformer_blocks.sublayers import AddAndNormWithDropoutLayer, \
    PositionWiseFFNLayer


class GPTDecoderBlock(nn.Module):
    """
    The Decoder block of the GPT Transformer.

    A bit different than the Transformer decoder block. No encoder-decoder attention.
    """

    def __init__(self, size: int, self_attention: MultiHeadedAttention,
                 feed_forward: PositionWiseFFNLayer, dropout: float):
        """
        Args:
            size (int):
                This hyper-parameter is used for the BatchNorm layers.
            self_attention (MultiHeadedAttention):
                The multi-headed attention layer that is used for self-attention.
            feed_forward (PositionWiseFFNLayer):
                The feed forward layer applied after attention for further transformation.
            dropout (float):
                Hyper-parameter used in drop-out regularization in training.
        """
        super(GPTDecoderBlock, self).__init__()
        self._size = size
        # (1) self-attention
        self._self_attention = self_attention
        # (2) add + norm layer
        self._add_norm_layer_1 = AddAndNormWithDropoutLayer(size, dropout)
        # (3) FFN
        self._feed_forward = feed_forward
        # (4) add + norm layer
        self._add_norm_layer_2 = AddAndNormWithDropoutLayer(size, dropout)

    @property
    def size(self) -> int:
        """
        Returns:
            size (int): This hyper-parameter is used for the BatchNorm layers.
        """

        return self._size

    def forward(self, values: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Main function call for decoder block.
        maps input -> self_attn -> addNorm -> src_attn -> FFN -> addNorm.
        Args:
            values (torch.Tensor): Either embedding(tgt) or decoder[l-1] output.
            src_mask (torch.Tensor):
                Masking target so model doesn't see padding or next sequential values.
        """
        # first self-attention on the decoder[l-1] layer
        values = self._add_norm_layer_1(values, lambda x: self._self_attention(x, x, x, src_mask))
        values = self._add_norm_layer_2(values, self._feed_forward)
        return values


class GPTCompositeDecoder(nn.Module):
    """
    The decoder stack of the Transformer.
    Pass the input through N decoder blocks then Add+Norm layer the output.
    """

    def __init__(self, layer: GPTDecoderBlock, num_layers: int):
        """
        Args:
            layer (DecoderBlock):layer to be sequentially repeated num_layers times.
            num_layers (int): Number of layers in the decoder block.
        """
        super(GPTCompositeDecoder, self).__init__()
        self._layers = nn.ModuleList([deepcopy(layer)] * num_layers)
        self._add_norm = nn.BatchNorm1d(layer.size, momentum=None, affine=False)

    def forward(self, values: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Main function call for Decoder block.
        Takes in embedding(target), target_mask, source, source_mask, and encoder output.

        Args:
            values (torch.Tensor): Either embedding(tgt) or decoder[l-1] output.
            src_mask (torch.Tensor):
                Masking target so model doesn't see padding or next sequential values.
        Returns:
            decoder output to be used as input into final layer.
        """
        for layer in self._layers:
            values = layer(values, src_mask)
        return self._add_norm(values)
