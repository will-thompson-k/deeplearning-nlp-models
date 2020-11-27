from argparse import Namespace

import torch
import numpy as np
from nlpmodels.models import gpt
from nlpmodels.utils import utils
from nlpmodels.utils.elt.gpt_batch import GPTBatch
from nlpmodels.utils.vocabulary import NLPVocabulary
from tests.test_data import gpt_regression_test_data

utils.set_seed_everywhere()


def test_input_output_dims_gpt():

    test_1_args = Namespace(
        num_layers_per_stack=2,
        dim_model=64,
        dim_ffn=64*4,
        num_heads=2,
        block_size=None,
        dropout=0.1,
    )

    # mock dataset
    src_tokens = ["the", "cows", "jumped", "over", "the", "moon", "and", "hey", "the", "british", "are", "coming"]
    batch_size = 1
    test_1_args.block_size = len(src_tokens)-1
    vocab = NLPVocabulary.build_vocabulary(src_tokens)
    src_indices = torch.LongTensor([vocab.lookup_token(s) for s in src_tokens])
    data = GPTBatch(src_indices[:-1].unsqueeze(0), src_indices[1:].unsqueeze(0),1)

    model = gpt.GPT(len(vocab),
                    test_1_args.num_layers_per_stack, test_1_args.dim_model,
                    test_1_args.dim_ffn, test_1_args.num_heads, test_1_args.block_size,
                    test_1_args.dropout)
    # push through model
    y_hat = model(data)

    # assert all dimensions are correct
    assert y_hat.size() == torch.Size([batch_size, test_1_args.block_size, len(vocab)])


def test_regression_test_gpt():
    utils.set_seed_everywhere()

    test_2_args = Namespace(
        num_layers_per_stack=2,
        dim_model=64,
        dim_ffn=64*4,
        num_heads=2,
        block_size=None,
        dropout=0.1,
    )

    # mock dataset
    src_tokens = ["the", "cow", "jumped", "over", "the", "moon", "and", "hey", "the", "british", "are", "coming"]
    batch_size = 1
    test_2_args.block_size = len(src_tokens)-1
    vocab = NLPVocabulary.build_vocabulary(src_tokens)
    src_indices = torch.LongTensor([vocab.lookup_token(s) for s in src_tokens])
    data = GPTBatch(src_indices[:-1].unsqueeze(0), src_indices[1:].unsqueeze(0),1)

    model = gpt.GPT(len(vocab),
                    test_2_args.num_layers_per_stack, test_2_args.dim_model,
                    test_2_args.dim_ffn, test_2_args.num_heads, test_2_args.block_size,
                    test_2_args.dropout)
    # push through model
    y_hat = model(data)

    #expected output
    expected_output = gpt_regression_test_data.GPT_REGRESSION_TEST_DATA

    # assert y_hat is within eps
    eps = 1.e-4
    assert np.allclose(y_hat.data.numpy(), expected_output.data.numpy(), atol=eps)