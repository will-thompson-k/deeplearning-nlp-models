"""
This module contains the skipgram/word2vec Dataset and Dataloader.
"""
from typing import Tuple, Any, List

import numpy as np
from datasets import load_dataset  # hugging_faces
from torch.utils.data import DataLoader

from nlpmodels.utils.dataset import AbstractNLPDataset
from nlpmodels.utils.tokenizer import tokenize_corpus_basic
from nlpmodels.utils.utils import set_seed_everywhere
from nlpmodels.utils.vocabulary import NLPVocabulary


class SkipGramDataset(AbstractNLPDataset):
    """
    SkipGramDataset class for transforming and storing dataset for use in Skip-gram model.
    """

    def __init__(self, data: List):

        self.data = data

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, int]:

        input_word_token, context_word_token = self.data[idx]
        return input_word_token, context_word_token

    @staticmethod
    def get_skipgram_context(sentence_tokens: List, context_size: int, word_probas: np.array, train=True) -> List:
        """
                Class method to take list of tokenized text and convert into sub-sampled (input,context) pairs.
                Note that sub-sampling only happens on the training dataset (see Mikolov et al. for details).

                    Args:
                        train_text (list): list of tokenized data to be used to derive (input,context) pairs.
                        dictionary (NLPVocabulary): a dictionary built off of the training data to map tokens <-> idxs.
                        context_size (int): the window around each input word to derive context pairings.
                        train (bool): a "train" flag to indicate we want to sub-sample the training set.
                    Returns:
                        list of (input_idx, context_idx) pairs to be used for negative sampling loss problem.
        """
        data_partitions = []
        sentence_len = len(sentence_tokens)
        # calculate prob(w) per paper for common word_sampling, section 2.3 of paper
        # designed to randomly drop high occurring words -> higher prob, higher chance to discard
        # take input_word and provide the context for left and right
        for input_idx, input_word in enumerate(sentence_tokens):
            for context_idx in range(max(input_idx - context_size, 0), min(input_idx + context_size, sentence_len - 1)):
                if context_idx != input_idx:
                    if train:
                        # sub-sampling methodology for training
                        if np.random.rand() < word_probas[context_idx] or np.random.rand() < word_probas[input_idx]:
                            continue
                    data_partitions.append((input_word, sentence_tokens[context_idx]))
        return data_partitions

    @classmethod
    def get_target_context_data(cls, train_text: List, dictionary: NLPVocabulary, context_size: int,
                                train: bool) -> List:
        """
        Class method to take list of tokenized text and convert into sub-sampled (input,context) pairs.
        Note that sub-sampling only happens on the training dataset (see Mikolov et al. for details).

            Args:
                train_text (list): list of tokenized data to be used to derive (input,context) pairs.
                dictionary (NLPVocabulary): a dictionary built off of the training data to map tokens <-> idxs.
                context_size (int): the window around each input word to derive context pairings.
                train (bool): a "train" flag to indicate we want to sub-sample the training set.
            Returns:
                list of (input_idx, context_idx) pairs to be used for negative sampling loss problem.
        """
        train_data = []
        word_probas = dictionary.get_word_discard_probas()
        for tokens in train_text:
            tokens = [dictionary.lookup_token(x) for x in tokens]
            train_data.extend(cls.get_skipgram_context(tokens, context_size, word_probas, train))
        return train_data

    @classmethod
    def get_training_dataloader(cls, *args: Any) -> Tuple[DataLoader, NLPVocabulary]:
        """
        Class method to take transformed dataset and package in torch.Dataloader
        so that random batches could be used in training.

            Args:
                context_size (int): size of the window to derive context words
                thresh (float): a hyper-parameter to be used in frequent word sub-sampling
            Returns:
                (torch.Dataloader, NLPVocabulary) tuple to be used downstream in training.
        """
        context, thresh, batch_size = args

        train_data, vocab = cls.get_training_data(context, thresh)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader, vocab

    @classmethod
    def get_training_data(cls, *args: Any) -> Tuple[AbstractNLPDataset, NLPVocabulary]:
        """
        Class method to generate the training dataset (derived from hugging faces "ag_news").
        This method grabs the raw text, tokenizes and cleans up the data, generates a dictionary,
        and generates a sub-sampled (input,context) pair for training.

            Args:
                context_size (int): size of the window to derive context words
                thresh (float): a hyper-parameter to be used in frequent word sampling
            Returns:
                (NLPDataset,NLPVocabulary) tuple to be used downstream in training.
        """
        context_size, thresh = args
        # Using the Ag News data via Hugging Faces
        train_text = load_dataset("ag_news")['train']['text']
        train_text = tokenize_corpus_basic(train_text)
        dictionary = NLPVocabulary.build_vocabulary(train_text)
        # for sub-sampling
        dictionary.set_proba_thresh(thresh)
        train_data = cls.get_target_context_data(train_text, dictionary, context_size, train=True)
        return cls(train_data), dictionary

    @classmethod
    def get_testing_data(cls, *args):
        pass

    @classmethod
    def get_testing_dataloader(cls, *args):
        pass


if __name__ == '__main__':
    window = 5
    t = 10.e-20  # from original paper: 10.e-5
    set_seed_everywhere()
    train_data, train_dict = SkipGramDataset.get_training_data(window, t)
    print(f"train_size = {len(train_data)},dict_size={len(train_dict)}")
