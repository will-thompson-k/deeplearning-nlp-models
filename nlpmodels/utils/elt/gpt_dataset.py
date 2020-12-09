"""
This module contains the GPT Dataset and GPT Dataloaders for the GPT problem.
"""

from typing import Tuple, Any

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from nlpmodels.utils.elt.dataset import AbstractNLPDataset
from nlpmodels.utils.tokenizer import tokenize_corpus_basic
from nlpmodels.utils.vocabulary import NLPVocabulary


class GPTDataset(AbstractNLPDataset):
    """
    GPT class for transforming and storing dataset for use in GPT language model.

    Uses torchtext's WikiText2 dataset.
    """

    def __init__(self, data: torch.Tensor, vocab: NLPVocabulary, block_size: int):
        """
        Args:
            data (torch.Tensor): 1D tensor of integers to sample batches from.
            vocab (NLPVocabulary): Vocabulary. Not target/source this time.
            block_size (int): Size of context window.
        """

        self._data = data
        self._vocab = vocab
        self._block_size = block_size

    @property
    def data(self) -> torch.Tensor:
        """
        Returns:
            data (torch.Tensor): 1D tensor of integers to sample batches from.
        """

        return self._data

    def __len__(self) -> int:
        """
        Returns: size of dataset.
        """

        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): index of dataset slice to grab.
        Returns:
            Tuple of tensors (source,target) for that index.
        """
        # only grabbing full length tensors
        idx = min(len(self.data)-self._block_size-1, idx)
        # grab a chunk of (block_size + 1) from the data
        chunk = self.data[idx:idx + self._block_size + 1]
        # return 2 block_size chunks shifted by 1 index
        return chunk[:-1], chunk[1:]

    @classmethod
    def get_training_dataloader(cls, args: Any) -> Tuple[DataLoader, NLPVocabulary]:
        """
        Returns a pytorch::Dataloader object and vocabulary ready for model training.

        Args:
            args: Parameters for deriving training data.
        Returns:
            Tuple of Dataloader class, source and target dictionaries
        """
        batch_size = args.batch_size
        block_size = args.block_size # size of context window.
        train_data, vocab = cls.get_training_data(block_size)

        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

        return train_loader, vocab

    @classmethod
    def get_training_data(cls, block_size: int) -> Tuple[AbstractNLPDataset, NLPVocabulary]:
        """
        Returns the dataset class along with vocabulary object.

        Args:
            block_size (int): The size of the context window.
        Returns:
            Tuple of the dataset and dictionary.
        """
        # download the huggingfaces::wikitext language model development
        train_dataset = load_dataset("wikitext", 'wikitext-2-raw-v1')['train']
        # flatten the pyarrow chunks into one string
        train_dataset = [" ".join([str(x) for x in train_dataset._data[0]])]
        train_dataset = tokenize_corpus_basic(train_dataset, False)
        # hack: i'm going to only grab the first 300k examples. cause this is like > 1MM words
        # build vocabulary
        vocab = NLPVocabulary.build_vocabulary([train_dataset[0]])
        train_dataset = torch.LongTensor([vocab.token_to_idx[x] for x in train_dataset[0]])
        # we pass the dataset, vocab... Dataset will do the rest
        return cls(train_dataset, vocab, block_size), vocab

