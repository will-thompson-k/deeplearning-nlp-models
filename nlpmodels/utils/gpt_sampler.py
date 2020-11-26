"""
This module contains a sampler for sampling a model output. Useful for language models.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from nlpmodels.utils.gpt_batch import GPTBatch


# Make sure we don't update the gradient.
@torch.no_grad()
def sampler(model: nn.Module,
            data: GPTBatch,
            steps: int,
            block_size: int,
            do_sample: bool = False) -> torch.Tensor:
    """
    This is a sampler for examining the performance of our model.


    Takes a sequence of tokens, predicts the next one in the sequence using our model,
    then appends to the sequence and takes a time step into the future.

    Inspired by Kaparthy's minGPT utils::sample function.

    Args:
        model (nn.Module): The model to sample from.
        data (GPTBatch): The data to start the sequence off.
        steps (int): Number of time steps to do our calculation.
        block_size (int): Size of the context window.
        do_sample (bool): Sampling
    Returns:
        new decoded sequence x [batch_size,context]->[batch_size,context+steps]
    """

    # freeze the model from updating
    model.eval()
    while steps:
        # data should be [batch_size,context,vocab]. Let's check dim 1
        data_src = data.src if data.src.size(1) <= block_size else data.src[:, -block_size:]
        data = GPTBatch(data_src, None, 0)
        # grab the predictions
        y_hat = model(data)
        # pluck the y_hat at the final step after reading in the whole context window
        y_hat = y_hat[:, -1]
        # apply softmax to convert to probabilities
        probas = F.softmax(y_hat, dim=-1)
        # sample from the distribution or take the most likely
        if do_sample:
            index = torch.multinomial(probas, num_samples=1)
        else:
            _, index = torch.topk(probas, k=1, dim=-1)
        # append to the sequence and continue
        data.src = torch.cat((data.src, index), dim=1)

        steps -= 1

    return data
