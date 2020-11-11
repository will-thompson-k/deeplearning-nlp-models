import numpy as np
import torch
from nlpmodels.utils.transformer_batch import TransformerBatch


class GPTBatch(TransformerBatch):
    """
    GPT batch data class encapsulating src, tgt, and related masks.

    """

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor, pad:int):
        """
        Args:
            src (torch.Tensor): The source input of (batch_size,block_size).
            tgt (torch.Tensor): The target input of (batch_size,block_size).
            pad (int): pad index to identify the padding in each sequence.
        """
        super().__init__(src, tgt, pad)
        self.src = src  # normal source
        self.src_mask = (src != pad).unsqueeze(1)  # 3D tensor necessary for attention
        self.tgt = tgt # target sequence
        self.tgt_mask = self.make_std_mask(self.tgt, pad)  # make padding conditional on point in sequence