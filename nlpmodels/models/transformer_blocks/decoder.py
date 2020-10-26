import torch.nn as nn
from nlpmodels.models.transformer_blocks.sublayers import AddAndNormWithDropoutLayer, PositionWiseFFNLayer
from nlpmodels.models.transformer_blocks.attention import MultiHeadedAttention
from copy import deepcopy


class DecoderBlock(nn.Module):
    """
    The Decoder block of the Transformer.
    """
    def __init__(self, size: int, self_attention: MultiHeadedAttention, source_attention: MultiHeadedAttention,
                 feed_forward: PositionWiseFFNLayer, dropout: float):
        super(DecoderBlock, self).__init__()
        self._size = size
        # (1) self-attention
        self._self_attention = self_attention
        # (2) add + norm layer
        self._add_norm_layer_1 = AddAndNormWithDropoutLayer(size, dropout)
        # (3) source-attention
        self._source_attention = source_attention
        # (4) add + norm layer
        self._add_norm_layer_2 = AddAndNormWithDropoutLayer(size, dropout)
        # (5) FFN
        self._feed_forward = feed_forward
        # (6) add + norm layer
        self._add_norm_layer_3 = AddAndNormWithDropoutLayer(size, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):

        x = self._add_norm_layer_1(x, lambda x: self._self_attention(x, x, x, tgt_mask))
        x = self._add_norm_layer_2(x, lambda x: self._source_attention(x, memory, memory, src_mask))
        x = self._add_norm_layer_3(x, self._feed_forward)
        return x


class CompositeDecoder(nn.Module):
    """
        The decoder stack of the Transformer.
        Pass the input through N decoder blocks then Add+Norm layer the output.
        """
    def __init__(self, layer: DecoderBlock, num_layers: int):
        super(CompositeDecoder, self).__init__()
        self._layers = nn.ModuleList([deepcopy(layer)] * num_layers)
        self._add_norm = nn.BatchNorm1d(layer._size, momentum=None, affine=False)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self._layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self._add_norm(x)
