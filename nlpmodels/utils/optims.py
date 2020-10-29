from argparse import Namespace

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class NoamOptimizer(object):
    """
    Noam optimizer that implements the Noam Learning rate schedule mentioned in
    "Attention is all you need" (2017).

    Tunes the LR of optimizer class during training by doing the following:
    (1) During warm-up, the LR increases linearly.
    (2) Afterwards, the warm_up steps decreases ~ 1/sqrt(step_number).

    Derived in part from logic found in "Annotated Transformer": https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, dim_model: int, factor: float, warm_up: int, optimizer: Optimizer):
        """
        Args:
            dim_model (int): size of the latent/embedding space.
            factor (float): hyper-parameter used for scaling LR change.
            warm_up (int): number of steps to include in warm_up calculation.
            optimizer (torch.optim.optimizers.Optimizer): the optimizer class to be modified.
        """
        # hyper-parameters
        self._dim_model = dim_model
        self._factor = factor
        self._warm_up = warm_up
        self._optimizer = optimizer

        # initialize parameters
        self._step = 0
        self._rate = 0

    def step(self):
        """
        Main method called during training.
        """
        # change learning rate in optimizer
        self._step += 1
        self._rate = self.calc_lr(self._step)
        if self._optimizer is not None:
            for p in self._optimizer.param_groups:
                p['lr'] = self._rate
            # call optimizer's step function
            self._optimizer.step()

    def zero_grad(self):
        """
        Clears out the gradients.
        """
        self._optimizer.zero_grad()

    def calc_lr(self, step: int) -> float:
        """
        Implements the LR schedule described in Attention (2017).

        Returns:
            New learning rate as a function of step.
        """
        return self._factor * ((self._dim_model ** (-0.5)) *
                               min(step ** (-0.5), (step * self._warm_up ** (-1.5))))

    @classmethod
    def get_transformer_noam_optimizer(cls, args: Namespace, model: nn.Module):
        """
        Instantiate the Noam optimizer with hyper-parameters specified in Attention (2017).

        Args:
            args (Namespace): contains the hyper-parameters of the run.
            model (nn.Module): the model to be trained.

        Returns:
            The NoamOptimizer optimizer.
        """
        return cls(args.dim_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
