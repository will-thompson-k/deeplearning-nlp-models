"""
This module contains the abstract base class for our datasets.
"""

# pylint: disable=missing-docstring

from typing import List

from abc import abstractmethod, ABC

from torch.utils.data import Dataset

from nlpmodels.utils.vocabulary import NLPVocabulary


class AbstractNLPDataset(Dataset, ABC):
    """
    Abstract base class for dataset class.
    """

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_training_data(cls, *args):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_training_dataloader(cls, *args):
        raise NotImplementedError

    @staticmethod
    def padded_string_to_integer(token_list: List[List[str]],
                                 max_sequence_length: int,
                                 vocab: NLPVocabulary) -> List[List[int]]:
        """
        Take a sequence of (string) tokens and convert them to a padded set of integers.

        Args:
            token_list (List[List[str]]):
                List of tokens to be converted to indices.
            max_sequence_length (int):
                Maximum length of sequence for each target, source sequence.
            vocab (NLPVocabulary):
                Dictionary to look up indices for each token.
        Returns:
            Sequence of indicies with EOS and PAD indices.
        """

        integer_list = []

        for tokens in token_list:
            integers = [vocab.mask_index] * max_sequence_length
            # this allows for truncated sequences.
            # In some problems, we will explicitly through out
            # datapoints < max_sequence_length prior to this step.
            integers[:len(tokens)] = [vocab.lookup_token(x) for x in tokens][:len(integers)]
            # Adding in the EOS token if the sequence is not truncated.
            if len(tokens) < max_sequence_length:
                integers[len(tokens)] = vocab.eos_index
            integer_list.append(integers)

        return integer_list

    @classmethod
    def prune_vocab(cls, vocab: NLPVocabulary, prob_thresh: float) -> NLPVocabulary:
        """
        A simple method that reduces the dictionary of a corpus to be more manageable.

        Args:
            vocab (NLPVocabulary): The original dictionary.
            prob_thresh (float): threshold of word frequency over which to keep tokens.
        Returns:
            Pruned dictionary.
        """
        word_probas = vocab.get_word_frequencies()
        # special tokens have 0 word_counts
        # this is a hard-coded hyper-parameter
        keep_words = word_probas > prob_thresh
        idx_to_token = vocab.idx_to_token
        keep_tokens = []
        for idx, keep in enumerate(keep_words):
            if keep:
                keep_tokens.append(idx_to_token[idx])
        # re-build the dictionary
        vocab = NLPVocabulary.build_vocabulary([keep_tokens])
        return vocab
