"""
This module contains the Transformer Dataset and Dataloader.
"""
from typing import Tuple, Any, List, AnyStr

import io

from functools import partial

import torch
from torch.utils.data import DataLoader
from torchtext.utils import download_from_url, extract_archive

from nlpmodels.utils.elt.dataset import AbstractNLPDataset
from nlpmodels.utils.tokenizer import tokenize_corpus_basic
from nlpmodels.utils.vocabulary import NLPVocabulary


class TransformerDataset(AbstractNLPDataset):
    """
    Transformer class for transforming and storing dataset for use in Transformer model.

    Uses torchtext's  dataset.
    """

    def __init__(self, data: List, target_vocab: NLPVocabulary):
        """
        Args:
            data (List): List of source, target tuples to be used in training/eval.
            target_vocab (NLPVocabulary): Target vocabulary.
        """

        self._data = data
        self._target_vocab = target_vocab

    def __len__(self) -> int:
        """
        Returns:
            size of dataset.
        """

        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """

        Args:
            idx (int): index of dataset slice to grab.

        Returns:
            Tuple of tensors (source,target) for that index.
        """
        source_integers, target_integers = self._data[idx]

        return torch.LongTensor(source_integers), torch.LongTensor(target_integers)

    @classmethod
    def get_training_dataloader(cls, args: Any) -> Tuple[DataLoader, NLPVocabulary, NLPVocabulary]:
        """
        Take in a set of parameters, return a pytorch::dataloader ready for training.

        Args:
            args: Parameters for deriving training data.

        Returns:
            Tuple of Dataloader class, source and target dictionaries
        """
        batch_size = args.batch_size
        max_sequence_length = args.max_sequence_length
        train_data, vocab_source, vocab_target = cls.get_training_data(max_sequence_length)

        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

        return train_loader, vocab_source, vocab_target

    @classmethod
    def get_training_data(cls, max_sequence_length: int) -> Tuple[AbstractNLPDataset, NLPVocabulary, NLPVocabulary]:
        """
        Take in a set of parameters, return a pytorch::dataloader ready for training.

        Args:
            max_sequence_length (int): The max sequence length.
        Returns:
            Tuple of the dataset and source and target dictionaries.
        """
        # download the multi-30k data raw for language translation
        # inspired by: https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
        # didn't want to use the torchtext 0.8.0 interface
        url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
        train_urls = ('train.de.gz', 'train.en.gz')
        train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]

        # yes, we are using the English tokenizer on German.
        # It's a simple tokenizer.
        de_tokenizer = partial(tokenize_corpus_basic, removal=False)
        en_tokenizer = partial(tokenize_corpus_basic, removal=False)

        de_vocab = cls.build_vocab(train_filepaths[0], de_tokenizer)
        en_vocab = cls.build_vocab(train_filepaths[1], en_tokenizer)

        # grab token lists in both languages.
        train_text_source, train_text_target = cls.build_token_lists(train_filepaths, de_tokenizer, en_tokenizer)

        assert len(train_text_source) == len(train_text_target)

        # throw out any data points that are > max_length
        train_text_filtered = [x for x in zip(train_text_source, train_text_target)
                               if len(x[0]) <= max_sequence_length - 1 and len(x[1]) <= max_sequence_length - 1]

        train_text_source, train_text_target = zip(*train_text_filtered)

        dictionary_source = de_vocab
        dictionary_target = en_vocab

        # convert to into padded sequences of integers
        train_text_source = cls.padded_string_to_integer(train_text_source, max_sequence_length, dictionary_source)
        train_text_target = cls.padded_string_to_integer(train_text_target, max_sequence_length + 1, dictionary_target)

        return cls(list(zip(train_text_source, train_text_target)),
                   dictionary_target), dictionary_source, dictionary_target

    @staticmethod
    def build_vocab(filepath: AnyStr, tokenizer):
        """
        This is a static method that builds an NLPVocabulary object from a file provided.

        Args:
            filepath(Anystr): This is a string for the filepath to open to build the vocab.
            tokenizer(function): This is a function to convert a list of strings into tokens.

        Returns:
            A NLPVocabulary object built off this list.
        """
        vocab = NLPVocabulary()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                vocab.add_many(tokenizer([string_])[0])
        return vocab

    @staticmethod
    def build_token_lists(filepaths: List, de_tokenizer, en_tokenizer):
        """
        This is a staticmethod used to return the data from files in a tokenized format.

        Args:
            filepaths(List): A list of both source and target lists.
            de_tokenizer(function): The tokenizer function for German.
            en_tokenizer(function): The tokenizer function for English.

        Returns:
            A tuple containing the list of tokens for each example.
        """
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        de_tokens_list = []
        en_tokens_list = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_tokens = de_tokenizer([raw_de])[0]
            en_tokens = en_tokenizer([raw_en])[0]
            de_tokens_list.append(de_tokens)
            en_tokens_list.append(en_tokens)
        return de_tokens_list, en_tokens_list
