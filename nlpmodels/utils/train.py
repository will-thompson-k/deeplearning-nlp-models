import torch.optim as optim
from tqdm import tqdm
from argparse import Namespace
import torch.nn as nn
from torch.utils.data import DataLoader


class Word2VecTrainer(object):
    '''
    Trainer class for the word2vec model.
    '''

    def __init__(self,args: Namespace,model: nn.Module,train_data: DataLoader):
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