from argparse import Namespace

import torch

from nlpmodels.utils.gpt_dataset import GPTDataset


def test_training_dataloader_batchsize():
    args_test = Namespace(
        batch_size=100,
        block_size=10,
    )
    gpt_dataloader = GPTDataset.get_training_dataloader(args_test)
    batch = next(iter(gpt_dataloader[0]))
    src_batch, tgt_batch = batch
    assert src_batch.size() == torch.Size(
        [args_test.batch_size, args_test.block_size]) and tgt_batch.size() == torch.Size(
        [args_test.batch_size, args_test.block_size])
