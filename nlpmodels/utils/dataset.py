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
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @classmethod
    @abstractmethod
    def get_training_data(cls, *args):
        pass

    @classmethod
    @abstractmethod
    def get_training_dataloader(cls, *args):
        pass

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
