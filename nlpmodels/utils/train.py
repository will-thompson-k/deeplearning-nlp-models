from argparse import Namespace
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlpmodels.utils import optims, label_smoother, transformer_batch


class Word2VecTrainer:
    '''
    Trainer class for the word2vec model.
    '''

    def __init__(self, args: Namespace, model: nn.Module, train_data: DataLoader):
        """
         Args:
             args (Namespace): a class containing all the parameters associated with run
             model (nn.Module): a PyTorch model
             train_data (DataLoader): a data loader that provides batches for running
         """
        self._args = args
        self._model = model
        self._train_data = train_data

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._args.learning_rate)

    def run(self):
        """
        Main running function for training a model.
        """
        for epoch in range(self._args.num_epochs):

            self._model.train()

            pbar = tqdm(self._train_data)
            pbar.set_description("[Epoch {}]".format(epoch))

            # iterate over batches
            for data in pbar:
                # step 1. zero the gradients
                self._optimizer.zero_grad()

                # step 2. forward_prop, compute the loss
                # note: not considering accuracy as we are learning context.
                loss = self._model(data)

                # step 3. back_prop
                loss.backward()

                # step 4. use optimizer to take gradient step
                self._optimizer.step()

                # status bar
                pbar.set_postfix(loss=loss.item())

        print("Finished Training...")


class TransformerTrainer:
    '''
    Trainer class for the Transformer model.
    '''

    def __init__(self, args: Namespace, vocab_target_size: int, pad_index: int, model: nn.Module,
                 train_data: DataLoader):
        """
        Args:
             args (Namespace): a class containing all the parameters associated with run
             vocab_target_size (int): size of target dictionary
             pad_index (int): index of pad token.
             model (nn.Module): a PyTorch model
             train_data (DataLoader): a data loader that provides batches for running
         """
        self._args = args
        self._model = model
        self._train_data = train_data

        self._loss_function = label_smoother.LabelSmoothingLossFunction(vocab_size=vocab_target_size,
                                                                        padding_idx=pad_index,
                                                                        smoothing=args.label_smoothing)

        # Noam optimizer per the paper (varies LR of Adam optimizer as a function of step)
        self._optimizer = optims.NoamOptimizer.get_transformer_noam_optimizer(args, model)

    def run(self):
        """
        Main running function for training a model.
        """
        for epoch in range(self._args.num_epochs):

            self._model.train()

            pbar = tqdm(self._train_data)
            pbar.set_description("[Epoch {}]".format(epoch))

            # iterate over batches
            for data in pbar:
                # re-format data for Transformer model
                data = self._reformat_data(data)

                # step 1. zero the gradients
                self._optimizer.zero_grad()

                # step 2. forward_prop
                y_hat = self._model(data)

                # step 3. compute the loss
                # convert y_hat into size (batch_size*max_seq_length,target_vocab_size)
                loss = self._loss_function(y_hat.contiguous().view(-1, y_hat.size(-1)),
                                           data.tgt_y.contiguous().view(-1))

                # step 4. back_prop
                loss.backward()

                # step 5. use optimizer to take gradient step
                self._optimizer.step()

                # NOTE: Usually makes sense to measure the loss from the val set (and add early stopping).
                # For this experiment, just running on training set entirely as an example.

                # status bar
                pbar.set_postfix(loss=loss.item())

        print("Finished Training...")

    @staticmethod
    def _reformat_data(data: Tuple) -> transformer_batch.TransformerBatch:
        """

        Args:
            data (Tuple): The tuples of LongTensors to be converted into a Batch object.
        Returns:
            TransformerBatch object containing data for a given batch.
        """
        # (batch,seq) shape tensors
        source_integers, target_integers = data

        # return a batch object with src,src_mask,tgt,tgt_mask tensors
        batch_data = transformer_batch.TransformerBatch(source_integers, target_integers, pad=0)

        return batch_data
