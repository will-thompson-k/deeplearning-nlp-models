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
                      # each kernel will look at window_size
                      # along max_sequence_length across full dim_model
                      kernel_size=(window_size, dim_model),
                      # want h_in==h_out in convolution
                      # (works for odd sized kernels)
                      padding=((window_size-1)//2, 0)
                      )
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
        # (batch_size, channels_in=1, max_sequence_length, dim_model)
        conv_input = embeddings.unsqueeze(1)

        conv_stack = []
        for conv_layer in self._convs:
            # (2) apply convolution layer + RELU
            # (batch_size, channels_in=1, h_in=max_sequence_length, w_in=dim_model) ->
            # (batch_size, channel_out=num_filters, h_out=max_sequence_length, w_out=1)
            conv_output = F.relu(conv_layer(conv_input)).squeeze(3)

            # This isn't super important since we are pooling across dim2,
            # but I wanted to make sure I got the same padding working.
            assert conv_output.size(0) == data.size(0) and \
                   conv_output.size(2) == data.size(1)

            # (3) Apply max pool along h_out dimension
            # (batch_size, channel_out=num_filters, h_out=max_sequence_length) ->
            # (batch_size, num_filters)
            pooled_output = F.max_pool1d(conv_output,
                                         conv_output.size(2)).squeeze(2)
            # (batch_size, num_filters), add to stack
            conv_stack.append(pooled_output)

        # flatten along dim=1
        # [(batch_size, num_filters)]*len(window_sizes) ->
        # (batch_size, num_filters)*len(window_sizes))
        pooled_values = torch.cat(conv_stack, 1)

        # (4) apply usual drop-out
        pooled_values = self._dropout(pooled_values)

        # (5) final linear layer, get into terms we care about
        # (batch_size, num_filters*len(window_sizes)) ->
        # (batch_size, class_num)
        y_hat = self._final_linear(pooled_values)

        # note: don't need to apply softmax, that is handled in CE function
        return y_hat

