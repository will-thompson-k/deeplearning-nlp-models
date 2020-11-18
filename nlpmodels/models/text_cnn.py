"""
This module contains the composite text-cnn model run on pre-trained word embeddings.
"""

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """

    """
    def __init__(self,
                 vocab_size: int,
                 num_layers_per_stack: int,
                 dim_model: int,
                 dropout: float):
        """
        Args:
            vocab_size (int):
            num_layers_per_stack (int):
            dim_model (int):
            dropout (float):
        """
        super(TextCNN, self).__init__()

        # (1) load pre-trained embeddings
        self._embeddings = None
        # (2) calculate CNN layers
        self._convs = None
        # (3) apply max-pooling
        self._pools = None
        # (4) apply drop out
        self._dropout = nn.Dropout(dropout=dropout)
        # (5) put through final linear layer
        self._final_linear = nn.Linear(dim_model, vocab_size)
        # init weights, bring in pre-trained embeddings
        self._init_weights()
        # keep the weights static for the embeddings since they are pre-trained.
        self._embeddings.weight.requires_grad = False

    def _init_weights(self):
        """
        Initialize the parameters, load the embeddings.
        """

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        The main function call of the model.

        Args:
            data (torch.Tensor):
        Returns:
            a tensor of (batch_size, class_size) of probability predictions
        """

        embeds = self._embeddings(data)

        pass
