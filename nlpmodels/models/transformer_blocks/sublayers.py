import torch.nn as nn
from torch.autograd import Variable
import torch
import math


class AddAndNormWithDropoutLayer(nn.Module):
    """
    The Add+Norm (w/ Dropout) subLayer.
    """
    def __init__(self, size, dropout):
        super(AddAndNormWithDropoutLayer, self).__init__()
        self._norm = nn.BatchNorm1d(size, momentum=None, affine=False)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # print(x.shape)
        batch_normed = self._norm(x)
        transformed = sublayer(batch_normed)
        return x + self._dropout(transformed)


class PositionWiseFFNLayer(nn.Module):
    """
    Implements the position-wise FFN sublayer.
    """
    def __init__(self, dim_model: int, dim_ffn: int):
        super(PositionWiseFFNLayer, self).__init__()
        self._W1 = nn.Linear(dim_model, dim_ffn)
        self._relu = nn.ReLU()
        self._W2 = nn.Linear(dim_ffn, dim_model)

    def forward(self, x) -> torch.Tensor:
        """
        Applies max(0,x) (i.e. RelU) + convoluted linear layers.
        """
        return self._W2(self._relu(self._W1(x)))


class PositionalEncodingLayer(nn.Module):
    """
    Implements positional encoding to capture the time signal/ relative
    position of each token in a sequence. The functional form is through sin/cos(pos).
    Note that this is a fixed layer and does not have any learned parameters.

    It has the same dimension as the embedding layer, which captures relative semantic meaning.
    This allows it to be able to be summed together. After (positional encoding + embedding), the embedded space is
    followed by attention.
    """

    def __init__(self,  dim_model: int, dropout: float, max_length: int):
        super(PositionalEncodingLayer, self).__init__()
        self._dropout = nn.Dropout(p=dropout)

        # create the positional_encoding once upon instantiation.
        p = torch.zeros(max_length, dim_model)
        # create tensor (size,1) with dim_0 == [0,1,2,..seq_length]
        position = torch.arange(0., max_length).unsqueeze(1)
        # calculate values with step_size 2 in log space along the embedding dimension
        div_term = torch.exp(torch.arange(0., dim_model, 2) *
                             -(math.log(10000.0) / dim_model))
        # frequency decomposition for each position in the sequence. (position, wave_length)
        p[:, 0::2] = torch.sin(position * div_term) # dim 2i
        p[:, 1::2] = torch.cos(position * div_term) # dim 2i + 1

        # Adding to register buffer will prevent it from being added to model.parameters()
        self.register_buffer('p', p)

    def forward(self, x):
        # We are keeping the PE tensors fixed.
        x = x + Variable(self.p, requires_grad=False)
        return self._dropout(x)


class NormalizedEmbeddingsLayer(nn.Module):
    """
    Implements a "normalized" embedding layer, which takes
    an embedding and normalizes by the sqrt(dim_model).
    """
    def __init__(self, vocab_size: int, dim_model: int):
        super(NormalizedEmbeddingsLayer, self).__init__()
        self._embeddings = nn.Embedding(vocab_size, dim_model)
        self._dim_model = dim_model

    def forward(self, x):
        return self._embeddings(x) * math.sqrt(self._dim_model)