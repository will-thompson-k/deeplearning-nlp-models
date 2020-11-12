import torch
import torch.nn as nn
from torch.nn import functional as F


# Make sure we don't update the gradient.
@torch.no_grad()
def greedy_sampler(model: nn.Module,
                   x: torch.Tensor,
                   steps: int,
                   block_size: int,
                   do_sample: bool = False) -> torch.Tensor:
    """
    This is a greedy decoder/sampler for examining the performance of our model.


    Takes a sequence of tokens, predicts the next one in the sequence using our model,
    then appends to the sequence and takes a time step into the future.

    Inspired by Kaparthy's minGPT utils::sample function.

    Args:
        model (nn.Module): The model to sample from.
        x (torch.Tensor):
            Tensor that is [batch_size, context window]. The data to start the sequence off.
        steps (int): Number of time steps to do our calculation.
        block_size (int): Size of the context window.
        do_sample (bool): Sampling

    Returns:
        new decoded sequence x [batch_size,context]->[batch_size,context+steps]
    """

    # freeze the model from updating
    model.eval()
    for k in range(steps):
        # x should be [batch_size,context,vocab]. Let's check dim 1
        X = x if x.size(1) <= block_size else x[:, -block_size:]
        # grab the predictions
        yhat = model(X)
        # pluck the yhat at the final step after reading in the whole context window
        yhat = yhat[:, -1]
        # apply softmax to convert to probabilities
        probas = F.softmax(yhat, dim=-1)
        # sample from the distribution or take the most likely
        if do_sample:
            ix = torch.multinomial(probas, num_samples=1)
        else:
            _, ix = torch.topk(probas, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x