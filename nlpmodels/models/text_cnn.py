"""
This module contains the composite text-cnn model run on pre-trained word embeddings.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    A CNN-based text classification architecture that uses convolutional layers on top of
    embeddings in order to make predictions.

    Derived (mainly) from Kim (2014) "Convolutional Neural Networks for Sentence Classification".
    """
    def __init__(self,
                 vocab_size: int,
                 dim_model: int,
                 num_filters: int,
                 window_sizes: List,
                 num_classes: int,
                 dropout: float):
        """
        Args:
            vocab_size (int): size of the vocabulary.
            dim_model (int): size of the embedding space.
            num_filters (int): the number of convolution filters to use.
            window_sizes (List): the kernel size.
            dropout (float): hyper-parameter used in drop-out regularization in training.
        """
        super(TextCNN, self).__init__()

        # (1) input embeddings
        self._embeddings = nn.Embedding(vocab_size, dim_model)
        # (2) calculate CNN layers
        self._convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(window_size, dim_model),
                      padding=(window_size - 1, 0))
            for window_size in window_sizes])
        # (3) apply drop out
        self._dropout = nn.Dropout(dropout)
        # (4) put through final linear layer
        self._final_linear = nn.Linear(num_filters * len(window_sizes), num_classes)
        # init weights
        self._init_weights(dim_model)

    def _init_weights(self, dim_model: int):
        """
        Initializes the weights of the embeddings vectors.

        Here we set the embeddings to be U(a,b) distribution.

        Args:
            dim_model (int): embedding space size.
        """

        self._embeddings.weight.data.uniform_(-0.5 / dim_model,
                                              0.5 / dim_model)

        # TODO: Add pre-loaded embeddings option.

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        The main function call of the model.

        Args:
            data (torch.Tensor):
                a tensor of size (batch_size, max_sequence_length)
        Returns:
            a tensor of (batch_size, class_size) of probability predictions
        """
        # (1) get embeddings
        # (batch_size, max_sequence_length) ->
        # (batch_size,max_sequence_length, dim_model)
        embeddings = self._embeddings(data)

        # (batch_size, max_sequence_length, dim_model) ->
        # (batch_size, channels=1, max_sequence_length, dim_model)
        conv_input = embeddings.unsqueeze(1)

        # (2) Apply RELU between each conv layer, add to stack
        # (batch_size, 1, max_sequence_length, dim_model) ->
        # conv_stack ->
        # [(batch_size, channel_out=num_filters, max_sequence_length)]*len(window_sizes)
        conv_stack = [F.relu(conv_layer(conv_input)).squeeze(3) for conv_layer in self._convs]

        # (3) Apply max_pool on RELU outputs
        # [(batch_size, num_filters, max_sequence_length)]*len(window_sizes)
        # max_pool -> (batch_size, num_filters, 1)
        # [(batch_size, num_filters)]*len(window_sizes)
        pooled_values = [F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
                         for conv_output in conv_stack]

        # flatten along dim=1
        # [(batch_size, num_filters)]*len(window_sizes) ->
        # (batch_size, num_filters)*len(window_sizes))
        pooled_values = torch.cat(pooled_values, 1)

        # (4) apply usual drop-out
        pooled_values = self._dropout(pooled_values)

        # (5) final linear layer, get into terms we care about
        # (batch_size, num_filters*len(window_sizes)) ->
        # (batch_size, class_num)
        y_hat = self._final_linear(pooled_values)

        # note: don't need to apply softmax, that is handled in CE function
        return y_hat
