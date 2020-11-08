import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class AddAndNormWithDropoutLayer(nn.Module):
    """
    The is a residual (skip) connection layer,
    where we add+batch_norm (w/ dropout) a sub-layer in the
    encoder and decoder blocks.
    """

    def __init__(self, size: int, dropout: float):
        """

        Args:
            size (int):
                From the documentation "C from an expected input of size (N, C, L)".
                Our equivalent is (batch_size, max_seq_length,dim_model),
                hence we are passing in max_seq_length.
            dropout (float):
                hyper-parameter used in dropout regularization.
        """
        super(AddAndNormWithDropoutLayer, self).__init__()
        # setting affine = False prevents the batchnorm parameter from being updated.
        # setting momentum = None calculates a simple moving average.
        self._norm = nn.BatchNorm1d(size, momentum=None, affine=False)
        self._dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, sublayer) -> torch.Tensor:
        """
        This is the main method used in the AddNorm residual connection layer.

        Args:
            values (torch.Tensor): The matrix that is added in a residual connection.
            sublayer (function): The sub-layer to be batch_normed and added to the original input.
        Returns:
            transformed matrix output.
        """
        batch_normed = self._norm(values)
        transformed = sublayer(batch_normed)
        return values + self._dropout(transformed)


class PositionWiseFFNLayer(nn.Module):
    """
    Implements the position-wise FFN sublayer.
    This can be thought of as 2 convolution layers with kernel size 1.
    Note that in this implementation dim_ffn > dim_model.

    """

    def __init__(self, dim_model: int, dim_ffn: int,
                 activ_func: nn.Module = nn.ReLU()):
        """
        Args:
            dim_model (int):
                size of the input matrix,
                which also happens to be the size of the embedding.
            dim_ffn (int):
                size of the FFN hidden layer.
        """
        super(PositionWiseFFNLayer, self).__init__()
        self._W1 = nn.Linear(dim_model, dim_ffn)
        self._activ_func = activ_func
        self._W2 = nn.Linear(dim_ffn, dim_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Applies activation_function + linear layers.
        Transformer: max(0,values) (i.e. RelU)
        GPT: GELU.

        Args:
            values (torch.Tensor):
                input of size (batch_size, max_seq_length, dim_model).
        Returns:
            output matrix of same size as input.
        """
        return self._W2(self._activ_func(self._W1(values)))


class PositionalEncodingLayer(nn.Module):
    """
    Implements positional encoding to capture the time signal/ relative
    position of each token in a sequence. The functional form is through sin/cos(pos).
    Note that this is a fixed layer and does not have any learned parameters.

    It has the same dimension as the embedding layer, which captures relative semantic meaning.
    This allows it to be able to be summed together.
    After (positional encoding + embedding), the embedded space is
    followed by attention.

    Derived in part from logic found in "Annotated Transformer":
    https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, dim_model: int, dropout: float, max_length: int):
        """

        Args:
            dim_model (int):
                size of the input matrix, which also happens to be the size of the embedding.
            dropout (float):
                Hyper-parameter used in drop-out regularization in training.
            max_length (int):
                The size of the sequence (fixed size, padded).
        """
        super(PositionalEncodingLayer, self).__init__()
        self._dropout = nn.Dropout(p=dropout)

        # create the positional_encoding once upon instantiation.
        pos_encoding = torch.zeros(max_length, dim_model)
        # create tensor (size,1) with dim_0 == [0,1,2,..seq_length]
        position = torch.arange(0., max_length).unsqueeze(1)
        # calculate values with step_size 2 in log space along the embedding dimension
        div_term = torch.exp(torch.arange(0., dim_model, 2) *
                             -(math.log(10000.0) / dim_model))
        # frequency decomposition for each position in the sequence. (position, wave_length)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pos_encoding[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1

        # Adding to register buffer will prevent it from being added to model.parameters()
        self.register_buffer('p', pos_encoding)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        The main call of the positional encoder.

        Args:
            values (torch.Tensor): embedding matrix of size (batch_size,dim_model).
        Returns:
            output of same size (batch_size, dim_model).
        """
        # We are keeping the PE tensors fixed.
        values = values + Variable(self.p, requires_grad=False)
        return self._dropout(values)


class NormalizedEmbeddingsLayer(nn.Module):
    """
    Implements a "normalized" embedding layer, which takes
    an embedding and normalizes by the sqrt(dim_model).
    """

    def __init__(self, vocab_size: int, dim_model: int):
        """

        Args:
            vocab_size (int): The size of the target or source vocabulary.
            dim_model (int): The size of the embedding space.
        """
        super(NormalizedEmbeddingsLayer, self).__init__()
        self._embeddings = nn.Embedding(vocab_size, dim_model)
        self._dim_model = dim_model

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        The main method for the normalized embedding model.

        Args:
            values (torch.Tensor):
                input of (batch_size,max_seq_length) size matrix to
                use as lookup of embedding vectors for
                each value in a sequence.
        Returns:
            output embedding matrix of (batch_size, max_seq_length, dim_model) size
        """
        return self._embeddings(values) * math.sqrt(self._dim_model)
