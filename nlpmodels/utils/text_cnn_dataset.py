"""
This module contains the cnn-text Dataset and Dataloader.
"""
from typing import Tuple, Any, List

import torch
from torch.utils.data import DataLoader

from torchtext.experimental.datasets import IMDB

from nlpmodels.utils.dataset import AbstractNLPDataset
from nlpmodels.utils.tokenizer import tokenize_corpus_basic
from nlpmodels.utils.vocabulary import NLPVocabulary


class TextCNNDataset(AbstractNLPDataset):
    """
    Text-CNN dataset for the text-cnn problem.

    Uses torchtext's experimental::IMDB dataset (sentiment analysis).
    """

    def __init__(self, data: List, vocab: NLPVocabulary):
        """
        Args:
            data (List): List of labels, text tuples to be used in training/eval.
            vocab (NLPVocabulary): vocabulary.
        """

        self.data = data
        self._vocab = vocab

    def __len__(self) -> int:
        """
        Returns:
            size of dataset.
        """

        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Args:
            idx (int): index of dataset slice to grab.
        Returns:
            Tuple of tensors (target, text) for that index.
        """
        target, text = self.data[idx]

        return torch.LongTensor(text), torch.LongTensor(target)

    @classmethod
    def get_training_dataloader(cls, args: Any) -> Tuple[DataLoader, NLPVocabulary]:
        """
        Take in a set of parameters, return a pytorch::dataloader ready for training.

        Args:
            args: Parameters for deriving training data.
        Returns:
            Tuple of Dataloader class, source and target dictionaries
        """
        batch_size = args.batch_size
        max_sequence_length = args.max_sequence_length
        train_data, vocab = cls.get_training_data(max_sequence_length)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader, vocab

    @classmethod
    def get_training_data(cls, max_sequence_length: int) -> Tuple[AbstractNLPDataset, NLPVocabulary]:
        """
        Download training data from torchtext, put into normalized formats.

        Args:
            max_sequence_length (int): The max sequence length.
        Returns:
            Tuple of the dataset and source and target dictionaries.
        """
        # download the IMDB data from torchtext.experimental for sentiment analysis
        train_dataset, = IMDB(data_select='train')
        # note: targets are {0,1} and the data is not shuffled
        train_target, train_text = zip(*train_dataset.data)
        # tokenize the data using our tokenizer
        train_text = tokenize_corpus_basic(train_text, False)
        # throw out any data points that are > max_length
        train_text = [x for x in train_text if len(x) <= max_sequence_length - 1]
        # build our vocab on the stripped text
        vocab = NLPVocabulary.build_vocabulary(train_text)
        # convert to into padded sequences of integers
        train_text = cls.padded_string_to_integer(train_text, max_sequence_length, vocab)

        return cls(list(zip(train_target, train_text)), vocab), vocab