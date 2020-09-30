import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkipGramNSModel(nn.Module):
    """
    Word2VecModel ("Skip-Gram") with Negative Sampling Loss function.
    """

    def __init__(self, vocab_size: int, embedding_size: int, negative_sample_size: int, word_frequency: torch.Tensor):
        """
               Args:
                   vocab_size (int): size of the vocabulary
                   embedding_size (int): size of the embeddings
                   negative_sample_size (int): size of negative examples to be sampled for loss function
                   word_frequency (torch.Tensor): the word frequencies from the vocabulary (to be used for selecting negative
                   examples)
        """
        super(SkipGramNSModel, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._negative_sample_size = negative_sample_size  # "k"

        self.input_embedding = nn.Embedding(self._vocab_size, self._embedding_size, padding_idx=0)
        self.output_embedding = nn.Embedding(self._vocab_size, self._embedding_size, padding_idx=0)

        self.input_embedding.weight.data.uniform_(-0.5 / self._embedding_size, 0.5 / self._embedding_size)
        self.output_embedding.weight.data.uniform_(-0.5 / self._embedding_size, 0.5 / self._embedding_size)

        # transform suggested in paper for U(w0)
        wf = np.power(word_frequency, 0.75)
        wf = wf / wf.sum()
        self._word_frequency = wf

    def forward(self, data: tuple) -> torch.Tensor:
        """
        The normal forward propagate function, but we output loss not prediction.

                Args:
                   data (tuple): batches of (input_word, context_word)
                Returns:
                    0-d tensor with loss
        """
        input_word, context_word = data

        batch_size = context_word.shape[0]

        # Rather than negative sample in the construction of the dataset,
        # we find negative samples here directly from embedding
        negative_sample_indices = torch.multinomial(self._word_frequency,
                                                    batch_size * self._negative_sample_size, replacement=True)\
                                                    .view(batch_size,-1)
        # negative_sample_indices = torch.randint(0, self._vocab_size - 1,
        #                                         size=(batch_size, self._negative_sample_size))

        input_vectors = self.input_embedding(input_word).unsqueeze(2)
        output_vectors = self.output_embedding(context_word).unsqueeze(2)
        neg_output_vectors = self.output_embedding(negative_sample_indices).neg()

        # we are calculating loss on 1 target=1 binary classification + k target=0 binary classifications
        pos_loss = F.logsigmoid(torch.mul(output_vectors, input_vectors).squeeze()).mean(1)
        neg_loss = F.logsigmoid(torch.bmm(neg_output_vectors, input_vectors).squeeze()).view(-1, 1, self._negative_sample_size).sum(2).mean(1)

        return -torch.mean(pos_loss + neg_loss)

    def get_embeddings(self) -> torch.Tensor:
        """
        Returns the input_embeddings associated with each word.

        Note: output_embeddings would be have a different meaning w.r.t. to context.

                Returns:
                    (vocab_size,embedding_size) torch.Tensor of values
        """

        return self.input_embedding.weight.data