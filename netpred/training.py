from itertools import count
from logging import info

import torch

from .networks import net, optimiser, criterion, scheduler
from .config import PRINT_EVERY, TRAINING_SET, VALIDATION_SET
from .data import ProteinDataLoader


train_loader = ProteinDataLoader(TRAINING_SET)
test_loader = ProteinDataLoader(VALIDATION_SET)


def print_statistics(epoch, batch, loss):
    print(f'[E{epoch} : {batch:6d}] loss = {loss:.3f}')


def train():
    """Train the network in a full pass through the train set, yielding after every batch."""
    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimiser.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimiser.step()

        yield i, loss.item()


def supervise():
    """Run training, halting when learning rate gets too small to be useful."""
    max_accuracy = 0

    for epoch in count(start=1):
        running_loss = []

        for batch, loss in train():
            running_loss.append(loss)

            if batch and not batch % PRINT_EVERY:
                loss = sum(running_loss) / PRINT_EVERY
                print_statistics(epoch, batch, loss)
                running_loss.clear()

        # early stopping
        accuracy = validation_accuracy()
        if accuracy >= max_accuracy:
            max_accuracy = accuracy
            torch.save(net.state_dict(), net.pth_name)

        scheduler.step(accuracy)
        if optimiser.param_groups[0]['lr'] < 3.e-4:
            break


def validation_accuracy() -> float:
    """Calculate 3-state prediction accuracy (Q3) on the validation set.
    Adapted from <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data>.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            predicted = outputs.max(dim=1).indices
            correct += (predicted == labels).sum().item()
            total += len(labels)
    accuracy = correct / total

    info(f'Accuracy on validation set: {100 * accuracy:.1f}%')
    return accuracy
