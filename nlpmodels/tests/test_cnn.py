from argparse import Namespace

import torch
import numpy as np
from nlpmodels.models import text_cnn
from nlpmodels.utils import utils
from nlpmodels.utils.vocabulary import NLPVocabulary
from nlpmodels.tests.test_data import cnn_regression_test_data

utils.set_seed_everywhere()


def test_input_output_dims_gpt():
    test_1_args = Namespace(
        vocab_size=300,
        # Model hyper-parameters
        max_sequence_length=400,  # Important parameter. Makes a big difference on output.
        dim_model=3,  # embedding size I tried 300->50
        num_filters=100,  # output filters from convolution
        window_sizes=[3, 5],  # different filter sizes, total number of filters len(window_sizes)*num_filters
        num_classes=2,  # binary classification problem
        dropout=0.5,  # 0.5 from original implementation, kind of high
        # Training hyper-parameters
        num_epochs=3,  # 30 from original implementation
        learning_rate=1.e-4,  # chosing LR is important, often accompanied with scheduler to change
        batch_size=64  # from original implementation
    )

    # mock dataset
    src_tokens = torch.randint(0, test_1_args.vocab_size - 1,
                                 size=(test_1_args.batch_size,test_1_args.max_sequence_length))

    model = text_cnn.TextCNN(test_1_args.vocab_size,
                             test_1_args.dim_model,
                             test_1_args.num_filters,
                             test_1_args.window_sizes,
                             test_1_args.num_classes,
                             test_1_args.dropout)
    # push through model
    y_hat = model(src_tokens)

    # assert all dimensions are correct
    assert y_hat.size() == torch.Size([test_1_args.batch_size, test_1_args.num_classes])


def test_regression_test_cnn():
    utils.set_seed_everywhere()

    test_2_args = Namespace(
        vocab_size=300,
        # Model hyper-parameters
        max_sequence_length=200,  # Important parameter. Makes a big difference on output.
        dim_model=3,  # embedding size I tried 300->50
        num_filters=100,  # output filters from convolution
        window_sizes=[3, 5],  # different filter sizes, total number of filters len(window_sizes)*num_filters
        num_classes=2,  # binary classification problem
        dropout=0.5,  # 0.5 from original implementation, kind of high
        # Training hyper-parameters
        num_epochs=3,  # 30 from original implementation
        learning_rate=1.e-4,  # chosing LR is important, often accompanied with scheduler to change
        batch_size=64  # from original implementation
    )

    # mock dataset
    src_tokens = torch.randint(0, test_2_args.vocab_size - 1,
                               size=(test_2_args.batch_size, test_2_args.max_sequence_length))

    model = text_cnn.TextCNN(test_2_args.vocab_size,
                             test_2_args.dim_model,
                             test_2_args.num_filters,
                             test_2_args.window_sizes,
                             test_2_args.num_classes,
                             test_2_args.dropout)
    # push through model
    y_hat = model(src_tokens)

    #expected output
    expected_output = cnn_regression_test_data.CNN_REGRESSION_TEST_DATA

    # assert y_hat is within eps
    eps = 1.e-4
    assert np.allclose(y_hat.data.numpy(), expected_output.data.numpy(), atol=eps)
