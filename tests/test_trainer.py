from argparse import Namespace

import torch
import numpy as np
from nlpmodels.models import gpt, text_cnn, transformer, word2vec
from nlpmodels.utils import utils, train
from nlpmodels.utils.elt import skipgram_dataset, gpt_dataset, text_cnn_dataset, transformer_dataset


def test_word2vec_trainer_regression_test():

    utils.set_seed_everywhere()

    test_args = Namespace(
        # skip gram data hyper-parameters
        context_window_size=2,
        subsample_t=10.e-200,
        # Model hyper-parameters
        embedding_size=300,
        negative_sample_size=15,
        # Training hyper-parameters
        num_epochs=2,
        learning_rate=1.e-3,
        batch_size=4096,
    )
    train_dataloader, vocab = skipgram_dataset.SkipGramDataset.get_training_dataloader(test_args.context_window_size,
                                                                                       test_args.subsample_t,
                                                                                       test_args.batch_size)
    word_frequencies = torch.from_numpy(vocab.get_word_frequencies())

    model = word2vec.SkipGramNSModel(len(vocab), test_args.embedding_size,
                                     test_args.negative_sample_size, word_frequencies)

    trainer = train.Word2VecTrainer(test_args, model, train_dataloader, True)
    trainer.run()
    losses = trainer.loss_cache
    # last loss across initial epochs should be converging
    assert losses[0].data > losses[-1].data


def test_gpt_trainer_regression_test():

    utils.set_seed_everywhere()

    test_args = Namespace(
        # Model hyper-parameters
        num_layers_per_stack=2,
        dim_model=12,
        dim_ffn=48,
        num_heads=2,
        block_size=64,
        dropout=0.1,
        # Training hyper-parameters
        num_epochs=5,
        learning_rate=0.0,
        batch_size=64,
    )

    train_loader, vocab = gpt_dataset.GPTDataset.get_training_dataloader(test_args)
    model = gpt.GPT(vocab_size=len(vocab),
                    num_layers_per_stack=test_args.num_layers_per_stack,
                    dim_model=test_args.dim_model,
                    dim_ffn=test_args.dim_ffn,
                    num_heads=test_args.num_heads,
                    block_size=test_args.block_size,
                    dropout=test_args.dropout)
    trainer = train.GPTTrainer(test_args, vocab.mask_index, model, train_loader, vocab, True)
    trainer.run()
    losses = trainer.loss_cache
    # last loss across initial epochs should be converging
    assert losses[0].data > losses[-1].data


def test_text_cnn_trainer_regression_test():

    utils.set_seed_everywhere()

    test_args = Namespace(
        # Model hyper-parameters
        max_sequence_length=50,
        dim_model=128,
        num_filters=128,
        window_sizes=[3,5,7],
        num_classes=2,
        dropout=0.5,
        # Training hyper-parameters
        num_epochs=10,
        learning_rate=1.e-6,
        batch_size=64
    )

    train_loader, vocab = text_cnn_dataset.TextCNNDataset.get_training_dataloader(test_args)
    model = text_cnn.TextCNN(vocab_size=len(vocab),
                             dim_model=test_args.dim_model,
                             num_filters=test_args.num_filters,
                             window_sizes=test_args.window_sizes,
                             num_classes=test_args.num_classes,
                             dropout=test_args.dropout)

    trainer = train.TextCNNTrainer(test_args, vocab.mask_index, model, train_loader, vocab, True)
    trainer.run()
    losses = trainer.loss_cache
    # last loss across initial epochs should be converging
    assert losses[0].data >= losses[-1].data


def test_transformer_trainer_regression_test():

    utils.set_seed_everywhere()

    test_args = Namespace(
        # Model hyper-parameters
        num_layers_per_stack=2,
        dim_model=512,
        dim_ffn=2048,
        num_heads=8,
        max_sequence_length=20,
        dropout=0.1,
        # Label smoothing loss function hyper-parameters
        label_smoothing=0.1,
        # Training hyper-parameters
        num_epochs=3,
        learning_rate=0.0,
        batch_size=128,
    )

    train_dataloader, vocab_source, vocab_target = transformer_dataset.TransformerDataset.get_training_dataloader(test_args)
    vocab_source_size = len(vocab_source)
    vocab_target_size = len(vocab_target)
    model = transformer.Transformer(vocab_source_size, vocab_target_size,
                                    test_args.num_layers_per_stack, test_args.dim_model,
                                    test_args.dim_ffn, test_args.num_heads, test_args.max_sequence_length,
                                    test_args.dropout)
    trainer = train.TransformerTrainer(test_args, vocab_target_size, vocab_target.mask_index, model, train_dataloader, True)
    trainer.run()
    losses = trainer.loss_cache
    # last loss across initial epochs should be converging
    assert losses[0].data >= losses[-1].data

test_transformer_trainer_regression_test()