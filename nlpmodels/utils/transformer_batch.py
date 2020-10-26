import numpy as np
import torch


class TransformerBatch(object):
    """ Borrowed from Annotated Transformer """
    def __init__(self, src, trg, pad=0):
        self.src = src # normal source
        self.src_mask = (src != pad).unsqueeze(1) # 3D tensor necessary for attention
        self.tgt = trg[:,:-1] # prev value of y
        self.tgt_y = trg[:,1:] # target (y)
        self.tgt_mask = self.make_std_mask(self.tgt, pad) # make padding conditional on point in sequence

    @classmethod
    def make_std_mask(cls, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(1) # 3D tensor necessary for attention
        seq_size = tgt.size(1)
        tgt_mask = tgt_mask & cls.subsequent_mask(seq_size) # subsequent mask built off sequence size
        return tgt_mask # batch_size, seq_length, seq_length

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        # create lower right triangle of 1s in numpy 3D tensor
        subsequent_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        # return as torch tensor
        return torch.from_numpy(subsequent_mask) == 0