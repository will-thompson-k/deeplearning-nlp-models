"""
This module contains the composite word2vec model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNSModel(nn.Module):
    """
    Word2VecModel ("Skip-Gram") with Negative Sampling Loss function.

    This is a self-supervised model where (input, context) pairs of words
    are trained to be "close" to each other within their projected embedding space.
    This can be thought of as a member of the "metric learning" class of problems.

    The brilliant insight here is rather than treat the problem as a typical
    softmax prediction, it is using the negative sampling loss function we are treating it as
    k+1 concurrent binary classification problems.

    This model is presented as the foundation for appreciating the power fo unsupervised learning
    in NLP, as we will see in later models as well (read: GPT,etc).
    """

    def __init__(self, vocab_size: int, embedding_size: int, negative_sample_size: int,
                 word_frequency: torch.Tensor):
        """
           Args:
               vocab_size (int): size of the vocabulary
               embedding_size (int): size of the embeddings
               negative_sample_size (int): size of negative examples to be sampled for loss function
               word_frequency (torch.Tensor): the word frequencies from the vocabulary
               (to be used for selecting negative examples)
        """
        super(SkipGramNSModel, self).__init__()

        # model hyper-parameters
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._negative_sample_size = negative_sample_size  # "k"

        # (1) input embedding (input word)
        self._input_embedding = nn.Embedding(vocab_size, embedding_size)
        # (2) output embedding (context word, sampled from context window)
        self._output_embedding = nn.Embedding(vocab_size, embedding_size)

        # word frequencies used in negative sampling loss
        # transform suggested in paper for U(w0)
        word_frequency = np.power(word_frequency, 0.75)
        word_frequency = word_frequency / word_frequency.sum()
        self._word_frequency = word_frequency

        # Init weights
        self._init_weights()

        self._device = 'cpu'
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()

    def _init_weights(self):
        """
        Initializes the weights of the embeddings vectors.

        Here we set the embeddings to be U(a,b) distribution.
        """

        self._input_embedding.weight.data.uniform_(-0.5 / self._embedding_size,
                                                   0.5 / self._embedding_size)

        self._output_embedding.weight.data.uniform_(-0.5 / self._embedding_size,
                                                    0.5 / self._embedding_size)

    def forward(self, data: tuple) -> torch.Tensor:
        """
        The main call function of the model, but we output loss not prediction.

        Args:
           data (tuple): batches of (input_word, context_word)
        Returns:
            0-d tensor with loss
        """

        # unpack of input, context vectors
        input_word, context_word = data

        batch_size = context_word.size(0)

        # (1) Grab negative sample indices (batch_size, k)
        # Rather than negative sample in the construction of the dataset,
        # we find negative samples here directly from embedding matrix
        # Note: the word_frequencies are set to 0 for special tokens and therefore
        # will not be sampled.
        negative_sample_indices = torch.multinomial(self._word_frequency,
                                                    batch_size * self._negative_sample_size,
                                                    replacement=True).view(batch_size, -1)

        negative_sample_indices = negative_sample_indices.to(self._device)

        # Alternative: If no information on word frequencies, sample uniformly.
        # negative_sample_indices = torch.randint(0, self._vocab_size - 1,
        #                                         size=(batch_size, self._negative_sample_size))

        # (2) Look-up input, context word embeddings
        # (batch_size, 1) -> (batch_size, 1, embedding_size) ->
        # (batch_size, embedding_size)
        input_vectors = self._input_embedding(input_word).unsqueeze(2)
        output_vectors = self._output_embedding(context_word).unsqueeze(2)

        # (3) Look-up negative context word embeddings
        # (batch_size, k) -> (batch_size, k, embedding_size)
        neg_output_vectors = self._output_embedding(negative_sample_indices).neg()

        # (4) Calculate loss function
        # 1 target=1 binary classification + k target=0 binary classifications
        pos_loss = F.logsigmoid(torch.mul(output_vectors, input_vectors).squeeze()).mean(1)
        neg_loss = F.logsigmoid(torch.bmm(neg_output_vectors, input_vectors).squeeze())\
            .view(-1, 1, self._negative_sample_size).sum(2).mean(1)

        # (5) Return collapsed loss functions
        return -torch.mean(pos_loss + neg_loss)

    def get_embeddings(self) -> torch.Tensor:
        """
        Returns the input_embeddings associated with each word.
        Note: output_embeddings would be have a different meaning w.r.t. to context.

        Returns:
            (vocab_size,embedding_size) torch.Tensor of values
        """

        return self._input_embedding.weight.data
