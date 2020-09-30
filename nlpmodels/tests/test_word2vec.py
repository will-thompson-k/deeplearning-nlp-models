from nlpmodels.models import word2vec
from argparse import Namespace
import torch
import numpy as np
from nlpmodels.utils import utils
utils.set_seed_everywhere()


def test_input_output_dims():

    test_1_args = Namespace(
        # Model hyper-parameters
        embedding_size=300,
        negative_sample_size=20,  # k examples to be used in negative sampling loss function
        # Training hyper-parameters
        batch_size=4096,
        # Vocabulary
        vocab_size = 1000,

    )
    word_frequencies = torch.from_numpy(np.random.rand(1000))
    mock_input_1 = torch.randint(0, test_1_args.vocab_size - 1,
                                                size=(test_1_args.batch_size,))
    mock_input_2 = torch.randint(0, test_1_args.vocab_size - 1,
                                                size=(test_1_args.batch_size,))
    data = (mock_input_1,mock_input_2)
    model = word2vec.SkipGramNSModel(test_1_args.vocab_size, test_1_args.embedding_size,
                                     test_1_args.negative_sample_size,word_frequencies)
    y = model(data)
    assert y.nelement() == 1


def test_embedding_size():

    test_2_args = Namespace(
        # Model hyper-parameters
        embedding_size=300,
        negative_sample_size=20,  # k examples to be used in negative sampling loss function
        # Training hyper-parameters
        batch_size=4096,
        # Vocabulary
        vocab_size=1000,

    )
    word_frequencies = torch.from_numpy(np.random.rand(1000))
    model = word2vec.SkipGramNSModel(test_2_args.vocab_size, test_2_args.embedding_size,
                                     test_2_args.negative_sample_size, word_frequencies)
    assert model.get_embeddings().shape == (test_2_args.vocab_size,test_2_args.embedding_size)
