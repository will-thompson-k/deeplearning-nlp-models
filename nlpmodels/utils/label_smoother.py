import torch
import torch.nn as nn
from torch.autograd import Variable


class LabelSmoothingLossFunction(nn.Module):
    """
    Implementation of label smoothing,
    a technique useful to improve robustness in multi-class classification
    that uses soft max.

    Rather than making the target a specific token_index,
    this loss function creates a probability distribution
    for each observation centered at the target and employs KL Divergence
    to minimize the cross entropy between
    the 2 distributions (predicted probabilites and "smoothed" y distribution).

    Attention (2017) uses smoothing hyper-param == 0.1.

    Design inspired by "Annotated Transformer":
    https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        """
        Args:
            vocab_size (int): Size of target dictionary.
            padding_idx (int): Index of padding token in the target dictionary.
            smoothing (float): hyper-parameter used in smoothing.
        """
        super(LabelSmoothingLossFunction, self).__init__()
        self._criterion = nn.KLDivLoss(size_average=False)
        self._padding_idx = padding_idx
        self._smoothing = smoothing
        self._vocab_size = vocab_size

    def forward(self, yhat: torch.Tensor, target: torch.Tensor) -> Variable:
        """
        Main function call of label smoother.
        Args:
            yhat (torch.Tensor):
                A sequence of (batch_size*max_seq_length,target_vocab_size) probability values.
            target (torch.Tensor):
                A 1D sequence of (batch_size) indicating the token values of the target (one-hot-encoding).

        Returns:
            Variable object containing the loss function.
        """

        # produces a target_smooth distribution that is (batch_size*max_seq_length,target_vocab_size)
        true_dist = self._compute_label_smoothing(target, yhat)

        # return KL divergence
        return self._criterion(yhat, Variable(true_dist, requires_grad=False))

    def _compute_label_smoothing(self, target: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        """

        Args:
            yhat (torch.Tensor):
                A sequence of (batch_size*max_seq_length,target_vocab_size) probability values.
            target (torch.Tensor):
                A 1D sequence of (batch_size) indicating the token values of the target (one-hot-encoding).

        Returns:
            a target_smooth distribution that is (batch_size*max_seq_length,target_vocab_size)

        """
        assert yhat.size(1) == self._vocab_size

        # generate matrix of the same distribution as yhat (copy)
        true_dist = yhat.data.clone()
        # fill with the smoothing values / vocab_size so that probabilities add to 1.0
        true_dist.fill_(self._smoothing / (self._vocab_size - 2))
        # along dimension 1, for the target indices (one-hot encoding indices), set to peak
        # of distribution which is 1.0 - smoothing_value
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self._smoothing)
        # make sure the padding index probability is == 0.0
        true_dist[:, self._padding_idx] = 0
        mask = torch.nonzero(target.data == self._padding_idx)
        true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return true_dist
