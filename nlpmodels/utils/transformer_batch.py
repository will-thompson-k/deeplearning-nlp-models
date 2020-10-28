import numpy as np
import torch


class TransformerBatch(object):
    """
    Transformer batch data class encapsulating src, tgt, and related masks.

    Borrowed from the "Annotated Transformer": https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """
    def __init__(self, src: torch.Tensor, tgt: torch.Tensor, pad: int = 0):
        """

        Args:
            src (torch.Tensor): The source input of (batch_size,max_seq_length).
            tgt (torch.Tensor): The target input of (batch_size,max_seq_length).
            pad (int): pad index to identify the padding in each sequence.
        """
        self.src = src # normal source
        self.src_mask = (src != pad).unsqueeze(1) # 3D tensor necessary for attention
        self.tgt = tgt[:,:-1] # prev value of y
        self.tgt_y = tgt[:,1:] # target (y)
        self.tgt_mask = self.make_std_mask(self.tgt, pad) # make padding conditional on point in sequence

    @classmethod
    def make_std_mask(cls, tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """
        Create a mask to hide padding and future words for target sequence.

        Args:
            tgt (torch.Tensor): The target input of (batch_size,max_seq_length) size.
            pad (int): pad index to identify the padding in each sequence.

        Returns:
            output matrix of size (batch_size,max_seq_length,max_seq_length) size with masked values for target sequence [True,False]
        """
        tgt_mask = (tgt != pad).unsqueeze(1) # 3D tensor necessary for attention, add in the middle.
        seq_size = tgt.size(1)
        tgt_mask = tgt_mask & cls.subsequent_mask(seq_size) # subsequent mask built off sequence size
        return tgt_mask

    @staticmethod
    def subsequent_mask(size: int) -> torch.Tensor:
        """
        Mask out subsequent positions. Used for target mask.

        Args:
            size (int): max_seq_length of each target sequence.

        Returns:
            output matrix of size (batch_size,max_seq_length,max_seq_length) size with masked values to ignore previous value.

        """
        # create lower right triangle of 1s in numpy 3D tensor
        subsequent_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        # return as torch tensor
        return torch.from_numpy(subsequent_mask) == 0