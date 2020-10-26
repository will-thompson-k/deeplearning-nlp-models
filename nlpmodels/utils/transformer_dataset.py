from torch.utils.data import DataLoader
from nlpmodels.utils.dataset import AbstractNLPDataset
from nlpmodels.utils.tokenizer import tokenize_corpus_basic
from nlpmodels.utils.vocabulary import NLPVocabulary
from typing import Tuple,Any,List
from torchtext.experimental.datasets import Multi30k
from argparse import Namespace
import numpy as np
import torch
from torch.autograd import Variable


class TransformerDataset(AbstractNLPDataset):
    """
    Transformer class for transforming and storing dataset for use in Transformer model.
    """
    def __init__(self, data: List, target_vocab: NLPVocabulary):

        self.data = data
        self._target_vocab = target_vocab

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor,torch.LongTensor]:
        # need to generate data.src, data.trg, data.src_mask, data.trg_mask
        source_integers, target_integers = self.data[idx]

        return torch.LongTensor(source_integers), torch.LongTensor(target_integers)

    @classmethod
    def get_training_dataloader(cls,args: Any) -> Tuple[DataLoader,NLPVocabulary,NLPVocabulary]:

        batch_size = args.batch_size
        max_sequence_length = args.max_sequence_length
        train_data, vocab_source, vocab_target = cls.get_training_data(max_sequence_length)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader,vocab_source,vocab_target

    @staticmethod
    def padded_string_to_integer(token_list: List[List[str]], max_sequence_length: int, vocab: NLPVocabulary) -> List[List[int]]:
        """
        Args:
            token_list (List[List[str]]):
            max_sequence_length (int):
            vocab (NLPVocabulary):

        Returns:
        """

        integer_list = []

        for tokens in token_list:
            integers = [vocab.mask_index]*max_sequence_length
            integers[:len(tokens)] = [vocab.lookup_token(x) for x in tokens]
            integers[len(tokens)] = vocab.eos_index
            integer_list.append(integers)

        return integer_list

    @classmethod
    def get_training_data(cls,max_sequence_length: int) -> Tuple[AbstractNLPDataset,NLPVocabulary,NLPVocabulary]:
        """

        Args:
            max_sequence_length:

        Returns:

        """
        # download the multi-30k data from torchtext.experimental for language translation
        tokenizers = (tokenize_corpus_basic,tokenize_corpus_basic)
        train_dataset, _, _ = Multi30k(tokenizer=tokenizers)
        # strip source (German) and target (English)
        train_text_source, train_text_target = zip(*train_dataset.data)
        # tokenize the data
        train_text_source = tokenize_corpus_basic(train_text_source,False)
        train_text_target = tokenize_corpus_basic(train_text_target,False)
        assert len(train_text_source) == len(train_text_target)
        # throw out any data points that are > max_length
        train_text_filtered = [x for x in zip(train_text_source, train_text_target)
                               if len(x[0]) <= max_sequence_length-1 and len(x[1]) <= max_sequence_length-1]
        train_text_source, train_text_target = zip(*train_text_filtered)
        # build source, target dictionaries
        dictionary_source = NLPVocabulary.build_vocabulary(train_text_source)
        dictionary_target = NLPVocabulary.build_vocabulary(train_text_target)
        # convert to into padded sequences of integers
        train_text_source = cls.padded_string_to_integer(train_text_source,max_sequence_length,dictionary_source)
        train_text_target = cls.padded_string_to_integer(train_text_target,max_sequence_length+1,dictionary_target)

        return cls(list(zip(train_text_source, train_text_target)),dictionary_target),dictionary_source,dictionary_target

    @classmethod
    def get_testing_data(cls, *args):
        pass

    @classmethod
    def get_testing_dataloader(cls, *args):
        pass


if __name__ == '__main__':
    args_test = Namespace(
        batch_size = 100,
        max_sequence_length = 10,
        )
    # y,_,_ = TransformerDataset.get_training_data(max_sequence_length=10)
    y,_,_ = TransformerDataset.get_training_dataloader(args_test)
    print(f"tested transformer dataset successfully")