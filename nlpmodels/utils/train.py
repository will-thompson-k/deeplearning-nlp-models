"""
This module contains all the trainers used for the different models in this repo.
Including:
+ AbstractTrainer
+ Word2vecTrainer
+ TransformerTrainer
+ GPTTrainer
+ TextCNNTrainer
"""

from argparse import Namespace
from typing import Tuple, List, Any
from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlpmodels.utils import optims, label_smoother, vocabulary
from nlpmodels.utils.elt import transformer_batch, gpt_batch


class AbstractTrainer(ABC):
    """
    Abstract base class for model trainers.
    """
    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 train_data: DataLoader,
                 loss_function: Any,
                 optimizer: torch.optim.Optimizer,
                 vocab: Any,
                 calc_accuracy: bool,
                 debug: bool):

        self._args = args
        self._model = model
        self._train_data = train_data
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._vocab = vocab
        self._calc_accuracy = calc_accuracy
        # for debugging
        self._loss_cache = []
        self._debug = debug
        self._iter = 0
        self._max_iter = 10

        # this asks DataParallel to
        # handle the distribution across gpus
        # potentially will swap out with DistributedDataParallel
        self._device = 'cpu'
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
            self._model = torch.nn.DataParallel(self._model).to(self._device)

    @property
    def loss_cache(self) -> List:
        """
        Returns:
            list of losses cached at the end of every epoch.
        """

        return self._loss_cache

    @abstractmethod
    def _reformat_data(self, data: Tuple) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _calc_loss_function(self, y_hat: Any, data: Any) -> Any:
        raise NotImplementedError

    def run(self):
        """
        Main running function for training a model.
        """
        for epoch in range(self._args.num_epochs):

            self._model.train()

            pbar = tqdm(self._train_data)
            pbar.set_description("[Epoch {}]".format(epoch))

            self._iter = 0

            correct = 0.0

            # iterate over batches
            for data in pbar:
                # re-format data for Transformer model
                data = self._reformat_data(data)

                # step 1. zero the gradients
                self._optimizer.zero_grad()

                # step 2. forward_prop
                y_hat = self._model(data)

                # step 3. compute the loss
                loss = self._calc_loss_function(y_hat, data)

                loss = loss.mean()

                # step 4. back_prop
                loss.backward()

                # step 5. use optimizer to take gradient step
                self._optimizer.step()

                # NOTE: Usually makes sense to measure the loss from the val set (and add early stopping).
                # For this experiment, just running on training set entirely as an example.

                if self._calc_accuracy:
                    target, text = data

                    correct += int((torch.max(y_hat, 1)[1].view(-1) == target.view(-1)).sum())

                    # status bar
                    pbar.set_postfix(loss=loss.item(), accuracy=100.0 * int(correct) / len(self._train_data.dataset))
                else:
                    # status bar
                    pbar.set_postfix(loss=loss.item())

                # debug
                if self._debug:
                    self._iter += 1
                    if self._iter >= self._max_iter:
                        break

            self._loss_cache.append(loss)

        print("Finished Training...")


class Word2VecTrainer(AbstractTrainer):
    """
    Trainer class for the word2vec model.
    """

    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 train_data: DataLoader,
                 debug: bool = False):
        """
        Args:
            args (Namespace): a class containing all the parameters associated with run
            model (nn.Module): a PyTorch model
            train_data (DataLoader): a data loader that provides batches for running
            debug (bool): True/false flag for debugging
        """

        loss_function = None
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        vocab = None
        calc_accuracy = False
        super(Word2VecTrainer, self).__init__(args,
                                              model,
                                              train_data,
                                              loss_function,
                                              optimizer,
                                              vocab,
                                              calc_accuracy,
                                              debug)

    def _reformat_data(self, data: Tuple) -> Tuple:
        # place data on the correct device
        return data[0].to(self._device), data[1].to(self._device)

    def _calc_loss_function(self, y_hat: Any, data: Any):

        return y_hat


class TransformerTrainer(AbstractTrainer):
    """
    Trainer class for the Transformer model.

    Using NoamOptimizer + LabelSmoothing.
    """
    def __init__(self,
                 args: Namespace,
                 vocab_target_size: int,
                 pad_index: int,
                 model: nn.Module,
                 train_data: DataLoader,
                 debug: bool = False):
        """
        Args:
            args (Namespace): a class containing all the parameters associated with run
            vocab_target_size (int): size of target dictionary
            pad_index (int): index of pad token.
            model (nn.Module): a PyTorch model
            train_data (DataLoader): a data loader that provides batches for running
            debug (bool): True/false flag for debugging
         """

        loss_function = label_smoother.LabelSmoothingLossFunction(vocab_size=vocab_target_size,
                                                                  padding_idx=pad_index,
                                                                  smoothing=args.label_smoothing)
        # Noam optimizer per the paper (varies LR of Adam optimizer as a function of step)
        optimizer = optims.NoamOptimizer.get_transformer_noam_optimizer(args, model)
        vocab = None
        calc_accuracy = False
        super(TransformerTrainer, self).__init__(args,
                                                 model,
                                                 train_data,
                                                 loss_function,
                                                 optimizer,
                                                 vocab,
                                                 calc_accuracy,
                                                 debug)

    def _reformat_data(self, data: Tuple) -> transformer_batch.TransformerBatch:
        """
        Args:
            data (Tuple): The tuples of LongTensors to be converted into a Batch object.
        Returns:
            TransformerBatch object containing data for a given batch.
        """
        # (batch,seq) shape tensors
        source_integers, target_integers = data

        # place data on the correct device
        source_integers = source_integers.to(self._device)
        target_integers = target_integers.to(self._device)

        # return a batch object with src,src_mask,tgt,tgt_mask tensors
        batch_data = transformer_batch.TransformerBatch(source_integers, target_integers, pad=0)

        return batch_data

    def _calc_loss_function(self, y_hat: Any, data: Any):

        # convert y_hat into size (batch_size*max_seq_length,target_vocab_size)
        return self._loss_function(y_hat.contiguous().view(-1, y_hat.size(-1)),
                                   data.tgt_y.contiguous().view(-1))


class GPTTrainer(AbstractTrainer):
    """
    Trainer class for the GPT model.

    Instead of Cosine Decay scheduler, will reuse the Noam Optimizer.
    Also using the usual cross entropy loss.
    """
    def __init__(self,
                 args: Namespace,
                 pad_index: int,
                 model: nn.Module,
                 train_data: DataLoader,
                 vocab: vocabulary.NLPVocabulary,
                 debug: bool = False):
        """
        Args:
            args (Namespace): a class containing all the parameters associated with run
            pad_index (int): index of pad token.
            model (nn.Module): a PyTorch model
            train_data (DataLoader): a data loader that provides batches for running
            vocab (NLPVocabulary): vocab of model
            debug (bool): True/false flag for debugging
         """

        # Usual cross entropy loss function
        loss_function = nn.CrossEntropyLoss(ignore_index=pad_index)
        # Note: I am using the original transformer's optimizer rather than the cosine decay function
        optimizer = optims.NoamOptimizer.get_transformer_noam_optimizer(args, model)

        calc_accuracy = False
        super(GPTTrainer, self).__init__(args,
                                         model,
                                         train_data,
                                         loss_function,
                                         optimizer,
                                         vocab,
                                         calc_accuracy,
                                         debug)

    def _reformat_data(self, data: Tuple) -> gpt_batch.GPTBatch:
        """
        Args:
            data (Tuple): The tuples of LongTensors to be converted into a Batch object.
        Returns:
            GPTBatch object containing data for a given batch.
        """
        # (batch,seq) shape tensors
        source_integers, target_integers = data

        # place data on the correct device
        source_integers = source_integers.to(self._device)
        target_integers = target_integers.to(self._device)

        # return a batch object with src,src_mask,tgt,tgt_mask tensors
        batch_data = gpt_batch.GPTBatch(source_integers,
                                        target_integers,
                                        self._vocab.mask_index)

        return batch_data

    def _calc_loss_function(self, y_hat: Any, data: Any):

        return self._loss_function(y_hat.view(-1, y_hat.size(-1)), data.tgt.view(-1))


class TextCNNTrainer(AbstractTrainer):
    """
    Trainer class for the Text-CNN model.

    Will use Adam optimizer without a learning rate scheduler.
    """
    def __init__(self,
                 args: Namespace,
                 pad_index: int,
                 model: nn.Module,
                 train_data: DataLoader,
                 vocab: vocabulary.NLPVocabulary,
                 debug: bool = False):
        """
        Args:
            args (Namespace): a class containing all the parameters associated with run
            pad_index (int): index of pad token.
            model (nn.Module): a PyTorch model
            train_data (DataLoader): a data loader that provides batches for running
            vocab (NLPVocabulary): vocab of model
            debug (bool): True/false flag for debugging
         """

        # Usual cross entropy loss function
        loss_function = nn.CrossEntropyLoss(ignore_index=pad_index, size_average=False)
        # Note: I am just using Adam, no LR scheduler.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        calc_accuracy = True
        super(TextCNNTrainer, self).__init__(args,
                                             model,
                                             train_data,
                                             loss_function,
                                             optimizer,
                                             vocab,
                                             calc_accuracy,
                                             debug)

    def _reformat_data(self, data: Tuple) -> Tuple:

        return data[0].to(self._device), data[1].to(self._device)

    def _calc_loss_function(self, y_hat: Any, data: Any):

        return self._loss_function(y_hat, data[0].view(-1))

