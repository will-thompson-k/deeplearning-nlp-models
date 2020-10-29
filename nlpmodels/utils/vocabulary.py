import numpy as np
from typing import List


class NLPVocabulary(object):
    """
    A vocabulary class used to map tokens to indices and calculate word frequencies.
    Design inspired by https://github.com/joosthub/PyTorchNLPBook.
    """

    def __init__(self, mask_token: str = "<MASK>", unk_token: str = "<UNK>", eos_token: str = "<EOS>"):
        """
        Args:
            mask_token (str) : token name used for masking
            unk_token (str) : token used for unknown words
            eos_token (str) : token used for end-of-string (sequence models)

        """
        self._token_to_idx = {}
        self._idx_to_token = {}
        self._word_count = {}

        self.unk_token = unk_token
        self.mask_token = mask_token
        self.eos_token = eos_token
        self._proba_thresh = 0

        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = self.add_token(self.unk_token)
        self.eos_index = self.add_token(self.eos_token)

    def add_token(self, token: str) -> int:
        """
        Add token to dictionary, return index.
            Args:
                token (str) : token name used for masking
            Returns:
                return the index associated with token.
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
            self._word_count[index] += 1
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
            self._word_count[index] = 0
        return index

    def add_many(self, tokens: List[str]) -> List[int]:
        """
        Add token to dictionary, return index.
            Args:
                tokens (List[str]) : tokens to be added to dictionary
            Returns:
                return the indices associated with tokens.
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token: str) -> int:
        """
        Lookup index of token.
            Args:
                token (str) : token to be used as key
            Returns:
                return the index associated with token.
        """
        if token not in self._token_to_idx:
            return self.unk_index
        return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        """
        Lookup token of index.
            Args:
                index (int) : index to be used as key
            Returns:
                return the token associated with index.
        """
        if index not in self._idx_to_token:
            raise KeyError(f"the index {index} is not in the Vocabulary")
        return self._idx_to_token[index]

    def lookup_word_count(self, index: int) -> dict:
        """
        Lookup word count of index.
            Args:
                index (int) : index to be used as key
            Returns:
                return map of index to word count
        """
        if index not in self._word_count:
            raise KeyError(f"the index {index} is not in the Word Count")
        return self._word_count[index]

    def set_proba_thresh(self, thresh: float):
        """
        Add thresh for probability of discarding frequent words.
            Args:
                thresh (float) : used in Mikolov paper to calculate word probability
        """
        self._proba_thresh = thresh

    def get_word_discard_probas(self):
        """
        Provides word discard probabilities

            Returns:
                np.array of word probabilities
        """
        word_frequency = self.get_word_frequencies()
        word_probas = 1 - np.sqrt(self._proba_thresh / word_frequency)
        word_probas = np.clip(word_probas, 0, 1)
        return word_probas

    def get_word_frequencies(self) -> np.array:
        """
         Provides word frequencies

             Returns:
                 np.array of word frequencies
        """
        word_counts_by_index = np.array([self._word_count[index] for index in range(len(self._idx_to_token))])
        word_frequency = word_counts_by_index / sum(word_counts_by_index)
        word_frequency = np.where(word_frequency == 0, self._proba_thresh, word_frequency)
        return word_frequency

    def __len__(self):

        return len(self._token_to_idx)

    @classmethod
    def build_vocabulary(cls, data: List[List[str]]):
        """
        Class method to return vocabulary object
            Args:
                data (List[List[str]]): get list of tokens
            Returns:
                 return class
        """
        dictionary = cls()
        for tokens in data:
            # add tokens to maps
            dictionary.add_many(tokens)

        return dictionary
