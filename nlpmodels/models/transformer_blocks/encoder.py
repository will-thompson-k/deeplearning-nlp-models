import torch.nn as nn
from nlpmodels.models.transformer_blocks.sublayers import AddAndNormWithDropoutLayer,PositionWiseFFNLayer
from nlpmodels.models.transformer_blocks.attention import MultiHeadedAttention
from copy import deepcopy


class EncoderBlock(nn.Module):
    """
    The Encoder block of the Transformer.
    This is repeated N times inside the Encoder stack.
    """
    def __init__(self, size: int, attention_layer: MultiHeadedAttention, feed_forward: PositionWiseFFNLayer, dropout: float):
        super(EncoderBlock, self).__init__()
        # (1) attention
        self._attention_layer = attention_layer
        # (2) add + norm layer
        self._add_norm_layer_1 = AddAndNormWithDropoutLayer(size, dropout)
        # (3) FFN
        self._feed_forward = feed_forward
        # (4) add + norm layer
        self._add_norm_layer_2 = AddAndNormWithDropoutLayer(size, dropout)
        self._size = size

    def forward(self, x, mask):
        """
           Main function call for encoder block.
           maps x -> self_attn -> addNorm -> FFN -> addNorm
        """
        x = self._add_norm_layer_1(x, lambda x: self._attention_layer(x, x, x, mask))
        x = self._add_norm_layer_2(x, self._feed_forward)
        return x


class CompositeEncoder(nn.Module):
    """
    The encoder stack of the Transformer.
    Pass the input through N encoder blocks then Add+Norm layer the output.
    """
    def __init__(self, layer: EncoderBlock, num_layers: int):
        super(CompositeEncoder, self).__init__()
        self._layers = nn.ModuleList([deepcopy(layer)]*num_layers)
        self._add_norm = nn.BatchNorm1d(layer._size, momentum=None, affine=False)

    def forward(self, x, mask):
        for layer in self._layers:
            x = layer(x, mask)
        return self._add_norm(x)

