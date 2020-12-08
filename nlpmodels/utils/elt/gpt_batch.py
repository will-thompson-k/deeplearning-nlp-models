"""
This module contains the batch object for the GPT model.
"""

import numpy as np
import torch


class GPTBatch:
    """
    GPT batch data class encapsulating src, tgt, and related masks.

    Unlike the Transformer, there is no self-attention for the src,tgt
    data. This is a language model, where the next token is predicted
    from the previous context_window tokens.
    """

    def __init__(self, src: torch.Tensor,
                 tgt: torch.Tensor,
                 pad: int):
        """
        Args:
            src (torch.Tensor): The source input of (batch_size,block_size).
            tgt (torch.Tensor): The target input of (batch_size,block_size).
            pad (int): pad index to identify the padding in each sequence.
        """
        # normal source
        self._src = src
        # target sequence, shifted 1, will only be used in loss function
        self._tgt = tgt

        self._device = 'cpu'
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()

        # make padding conditional on point in sequence
        self._src_mask = self.make_std_mask(self._src, pad)

    @property
    def src(self) -> torch.Tensor:
        """
        Returns:
            src (torch.Tensor): The source input of (batch_size,block_size).
        """

        return self._src

    @property
    def tgt(self) -> torch.Tensor:
        """
        Returns:
            tgt (torch.Tensor): The target input of (batch_size,block_size).
        """

        return self._tgt

    @property
    def src_mask(self) -> torch.Tensor:
        """
        Returns:
            output matrix of size (batch_size,max_seq_length,max_seq_length)
            size with masked values for target sequence [True,False]
        """

        return self._src_mask

    def make_std_mask(self, tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Create a mask to hide padding and future words for src sequence.

        Args:
            tgt (torch.Tensor): The target input of (batch_size,context_window) size.
            pad (int): pad index to identify the padding in each sequence.

        Returns:
            output matrix of size (batch_size,max_seq_length,max_seq_length)
            size with masked values for target sequence [True,False]
        """
        # 3D tensor necessary for attention, add in the middle.
        tgt_mask = (tgt != pad).unsqueeze(1)
        seq_size = tgt.size(1)
        # subsequent mask built off sequence size
        tgt_mask = tgt_mask & self.subsequent_mask(seq_size).to(self._device)
        return tgt_mask

    @staticmethod
    def subsequent_mask(size: int) -> torch.Tensor:
        """
        Mask out subsequent positions. Used for target mask.

        Args:
            size (int): max_seq_length of each target sequence.
        Returns:
            output matrix of size (batch_size,max_seq_length,max_seq_length)
            size with masked values to ignore previous value.
        """
        # create lower right triangle of 1s in numpy 3D tensor
        subsequent_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        # return as torch tensor
        return torch.from_numpy(subsequent_mask) == 0
