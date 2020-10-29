import re
import string
from typing import List


class BasicEnglishTokenizer:
    """
    Tokenizer useful for normalizing words and generating tokens.
    (Design inspired by torchtext.data.utils )
    """
    _patterns = [r'\W',
                 r'\d+',
                 r'\s+', ]  # Remove all words < 4 letters? r'\b\w{,3}\b'

    _replacements = [' ',
                     ' ',
                     ' ']

    _patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

    def tokenize(self, line: str) -> List[str]:
        """
        Function to tokenize a string within tokenizer.

        Args:
            line (str): the line to be tokenized
        Returns:
            a list of tokens
        """
        line = line.lower()
        for pattern_re, replaced_str in self._patterns_dict:
            # apply regex substitutes
            line = pattern_re.sub(replaced_str, line)
            # remove underscores
            line = line.replace('_', '')
        # remove all 1 letters
        line = line.split()
        line = [x for x in line if len(x) > 1]
        return line


def tokenize_corpus_basic(text: List[str], removal: bool = True) -> List[List[str]]:
    """
    Function to tokenize a list of strings (corpus).

    Args:
        text (str): a list of strings to be converted into tokens.
        removal (bool): remove token list if it is empty
    Returns:
        a list of a list of tokens.
    """
    tokenizer = BasicEnglishTokenizer()
    new_text = []

    for line in text:
        # tokenize sentence
        tokens = [x for x in tokenizer.tokenize(line) if x not in string.punctuation]
        if not tokens and removal:
            continue
        new_text.append(tokens)

    return new_text
