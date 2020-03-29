# coding=utf-8

import torch


class BaseTorchTrainer:
    """Base Trainer for Torch Models."""
    def __init__(self, epochs, optimizer, model, criterion=None, scheduler=None,
                 verbose=False):
        """Initialization."""
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.verbose = verbose

    def train_step(self, iterator):
        """
        
        Args:
            iterator ():

        Returns:

        """
        self.model.train()
        epoch_loss = 0
        epcoh_acc = 0

        for batch in iterator:
            self.optimizer.zero_grad()
            proba = self.model(batch)
            loss = self.criterion(proba, batch.label)

            #acc = binary_accuracy(predictions, batch.label)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            #epoch_acc += acc.item()

        return epoch_loss / len(iterator)#, epoch_acc / len(iterator)
