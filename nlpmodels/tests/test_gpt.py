from argparse import Namespace

import torch

from nlpmodels.models import gpt
from nlpmodels.utils import utils
from nlpmodels.utils.gpt_batch import GPTBatch
from nlpmodels.utils.vocabulary import NLPVocabulary

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
    src_tokens = ["the", "cow", "jumped", "over", "the", "moon", "and", "hey", "the", "british", "are", "coming"]
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
    yhat = model(data)

    # assert all dimensions are correct
    assert yhat.size() == torch.Size([batch_size, test_1_args.block_size, len(vocab)])


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
    yhat = model(data)

    #expected output
    expected_output = torch.Tensor([[[-0.0097,  1.0404,  0.0174, -0.1330,  0.2909,  0.3238, -0.6621,
          -0.2410,  0.1851,  0.6474, -0.5480, -0.6960, -0.2905, -0.9209,
           0.5060, -0.3861, -0.2811, -0.2958,  0.3174,  0.1764,  0.3202,
           0.3465, -0.9749],
         [ 0.0650,  0.3883, -0.1118,  0.1712,  0.1720,  0.4591, -0.9180,
           0.0501,  0.1643,  0.7689, -0.2787, -1.0378, -0.4031, -0.7475,
           0.6456,  0.1966, -0.2435, -0.2278, -0.1136, -0.1853,  0.6346,
           0.4911, -0.5700],
         [-0.0515,  0.4492, -0.0839, -0.1854,  0.0792,  0.4848, -0.9012,
          -0.0810,  0.0318,  0.9985, -0.3362, -1.1412, -0.4469, -0.8763,
           0.8003,  0.1788, -0.0034, -0.1307, -0.1275, -0.2860,  0.8286,
           0.4082, -0.5836],
         [-0.0507,  0.5959, -0.1618,  0.0093,  0.1059,  0.3708, -0.8053,
           0.0045,  0.0536,  0.7624, -0.1332, -0.9862, -0.3482, -0.7883,
           0.5618, -0.0169, -0.1662, -0.1617, -0.0585, -0.0782,  0.4347,
           0.2619, -0.5523],
         [-0.7011,  0.5730, -0.4553,  0.3939,  0.7523,  0.6391, -0.8054,
           0.1306, -0.0232,  1.1693, -0.1988, -0.7461, -0.3815, -1.1176,
           0.8466,  0.1131, -0.2185,  0.0504,  0.0382, -0.0170,  1.1783,
           0.3803, -0.6980],
         [-0.0096,  0.2866, -0.0439,  0.4957,  0.5372,  0.1378, -0.9578,
           0.0552,  0.0566,  0.8820, -0.2430, -1.0028, -0.7269, -0.6891,
           0.7850,  0.4012, -0.0872, -0.2700,  0.1120, -0.2694,  0.3450,
           0.6063, -0.3308],
         [-0.1761,  0.8059, -0.1935,  0.1312,  0.2939,  0.3623, -0.7971,
           0.1781,  0.0817,  0.7541, -0.0948, -1.0886, -0.3922, -0.7117,
           0.5595,  0.0529, -0.2916, -0.1453,  0.0059, -0.1784,  0.4880,
           0.3580, -0.5543],
         [-0.1034,  0.2696,  0.1683,  0.0462, -0.0152,  0.3747, -0.8122,
          -0.3834,  0.2291,  1.1183, -0.0250, -1.1861, -0.3929, -0.7932,
           0.4826,  0.1984, -0.0824,  0.1174, -0.1494, -0.0254,  0.5444,
           0.4095, -0.2374],
         [-0.2247,  0.5856, -0.1867,  0.1161,  0.2140,  0.5044, -0.8987,
          -0.0571,  0.0041,  0.8660, -0.1414, -1.1018, -0.3131, -0.8884,
           0.5948,  0.1649, -0.1792, -0.2801, -0.0423, -0.0250,  0.7543,
           0.4938, -0.7045],
         [-0.4951,  0.6853, -0.4565,  0.3036,  0.5690,  0.5367, -0.9067,
          -0.1031, -0.2971,  1.0977, -0.4591, -0.9541, -0.2878, -0.9077,
           0.5642,  0.2528, -0.1009, -0.0054,  0.1238,  0.1191,  0.8477,
           0.6583, -0.3574],
         [ 0.0554,  0.3149,  0.0325,  0.3481,  0.2671,  0.2591, -0.9455,
          -0.1617, -0.0782,  0.8567, -0.2720, -1.1613, -0.4671, -0.7517,
           0.7278,  0.3387, -0.0800, -0.2795, -0.0730, -0.0258,  0.3536,
           0.6046, -0.4531]]])

    # assert yhat is within eps
    eps = 1.e-2
    assert torch.sum(yhat-expected_output).data.numpy() < eps