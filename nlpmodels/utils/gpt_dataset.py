from typing import Tuple, Any, List

import torch
from torch.utils.data import DataLoader
from torchtext.experimental.datasets import WikiText2

from nlpmodels.utils.dataset import AbstractNLPDataset
from nlpmodels.utils.tokenizer import tokenize_corpus_basic
from nlpmodels.utils.vocabulary import NLPVocabulary


class GPTDataset(AbstractNLPDataset):
    """
    GPT class for transforming and storing dataset for use in GPT language model.
    """

    def __init__(self, data: torch.Tensor, vocab: NLPVocabulary, block_size: int):
        """
        Args:
            data (torch.Tensor): 1D tensor of integers to sample batches from.
            vocab (NLPVocabulary): Vocabulary.
            block_size (int):
        """

        self.data = data
        self._vocab = vocab
        self._block_size = block_size

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
            Tuple of tensors (source,target) for that index.
        """
        # grab a chunk of (block_size + 1) from the data
        chunk = self.data[idx:idx + self._block_size + 1]
        # return 2 block_size chunks shifted by 1 index
        return torch.LongTensor(chunk[:-1]), torch.LongTensor(chunk[1:])

    @classmethod
    def get_training_dataloader(cls, args: Any) -> Tuple[DataLoader, NLPVocabulary]:
        """
        Args:
            args: Parameters for deriving training data.

        Returns:
            Tuple of Dataloader class, source and target dictionaries
        """
        batch_size = args.batch_size
        block_size = args.block_size # size of context window.
        train_data, vocab = cls.get_training_data(block_size)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader, vocab

    @classmethod
    def get_training_data(cls, block_size: int) -> Tuple[AbstractNLPDataset, NLPVocabulary]:
        """
        Args:
            block_size (int): The size of the context window.
        Returns:
            Tuple of the dataset and source and target dictionaries.
        """
        # download the WikiText2 data from torchtext.experimental for language model development
        # this dataset already is already tokenized and converted into integers.
        train_block, _, _ = WikiText2()
        # gran the actual data and the vocab
        train_dataset, train_vocab = train_block.data, train_block.vocab
        # convert the torchtext dictionary into our internal dictionary
        vocab = NLPVocabulary(unk_token=train_vocab.UNK,mask_token=train_vocab.itos[1])
        cls.convert_torchtext_vocab(train_vocab, vocab)
        # we pass the dataset, vocab... Dataset will do the rest
        return cls(train_dataset, vocab, block_size), vocab

    @classmethod
    def convert_torchtext_vocab(cls, train_vocab, vocab):
        # got to change the order of some defaults, Pytorch and I don't see eye to eye apparently
        vocab._token_to_idx[vocab.unk_token] = 0
        vocab._idx_to_token[0] = vocab.unk_token
        vocab.unk_index = 0
        vocab._token_to_idx[vocab.mask_token] = 1
        vocab._idx_to_token[1] = vocab.mask_token
        vocab.mask_index = 1
        # got to handle this <eos> token that is not found here.
        del vocab._token_to_idx[vocab.eos_token], vocab._idx_to_token[vocab.eos_index]
        del vocab._word_count
        vocab.eos_index = -1
        # add in the other tokens
        vocab._idx_to_token = dict([(i, x) for i, x in enumerate(train_vocab.itos)])
        vocab._token_to_idx = train_vocab.stoi

    @classmethod
    def get_testing_data(cls, *args):
        pass

    @classmethod
    def get_testing_dataloader(cls, *args):
        pass