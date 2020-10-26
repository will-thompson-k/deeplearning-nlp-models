import torch.nn as nn
import torch
from torch.autograd import Variable


class LabelSmoothingLossFunction(nn.Module):
    """
    Implementation of label smoothing, a technique useful to improve robustness in multi-class classification
    that uses soft max.

    Using KL Divergence, we take the one-hot encoding of the target labels and transform into a smooth probability
    distribution over the vocabulary with the peak at the index where y==1.

    Attention (2017) uses smoothing hyper-param == 0.1.
    """

    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        """
        Args:
            vocab_size (int):
            padding_idx (int):
            smoothing (float):
        """
        super(LabelSmoothingLossFunction, self).__init__()
        self._criterion = nn.KLDivLoss(size_average=False)
        self._padding_idx = padding_idx
        self._smoothing = smoothing
        self._vocab_size = vocab_size

    def forward(self, yhat, target):

        # yhat should be (max_seq_length,target_vocab_size)
        # y should be 1D max_seq_length
        assert yhat.size(1) == self._vocab_size
        true_dist = yhat.data.clone()
        true_dist.fill_(self._smoothing / (self._vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self._smoothing)
        true_dist[:, self._padding_idx] = 0
        mask = torch.nonzero(target.data == self._padding_idx)
        true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self._criterion(yhat, Variable(true_dist, requires_grad=False))