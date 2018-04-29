""" Code for calculating losses, training and testing neural network models

"""
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from utils import Hp


def classification_loss(model, data, target):
    """ Return loss for classification models

    Args:
        model:  Neural network model
        data:   Mini-batch input data
        target: Mini-batch target

    Returns:
        loss:   Loss on mini-batch

    """
    output = model(data)
    return F.nll_loss(output, target)


def rnn_loss(model, data, target):
    """ Return loss for rnn models

    Args:
        model:              Neural network model
        data [1 x d x t]:   Input sequence data, where d is the input dimension and t is the number of time periods
        target [1]:         Target

    Returns:
        loss:               Loss

    """
    # Get rid of zeroth dimension, since the minibatch is of size 1
    data = data[0, :, :]  # [d x t]

    hidden = model.initHidden()

    sequence_length = data.size()[1]
    for i in range(sequence_length):
        input = data[:, i]  # [d]
        output, hidden = model(input, hidden)
    return nn.MSELoss()(output, target)


def get_loss(model, data, target):
    """ Return loss for model depending on the type of data,
    which is specified in Hp.data_type

    Args:
        model:          Neural network model
        data:           Input data
        target:         Target

    Returns:
        loss:           Loss

    """
    if Hp.data_type == 'classification':
        return classification_loss(model, data, target)
    elif Hp.data_type == 'sequential':
        return rnn_loss(model, data, target)


# Define training
def train(model, train_loader, optimizer, epoch):
    """ Trains the neural network model

    Args:
        model:          Neural network model
        train_loader:   Class that loads mini-batches
        optimizer:      Optimization algorithm
        epoch:          Current epoch number

    """
    model.train()
    cum_loss = 0  # Cumulative loss between printing of training loss
    cum_iterations = 0  # Cumulative number of datapoints between printing training loss

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        loss = get_loss(model, data, target)
        loss.backward()

        # Record loss
        cum_loss += loss.data[0]
        cum_iterations += 1

        # Clip gradients
        # As implemented in https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L162-L164
        if Hp.clipping_threshold != 0:
            clip_grad_norm(model.parameters(), Hp.clipping_threshold)

        # Take optimization step
        optimizer.step()

        # Print loss on current datapoint
        if batch_idx % 1000 == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), cum_loss / cum_iterations))

            # Reset cumulative losses
            cum_loss = 0
            cum_iterations = 0


# Define testing
def test(model, test_loader):
    """ Tests the neural network model

    Args:
        model:          Neural network model
        test_loader:    Class that loads mini-batches

    Returns:
        prints average loss

    """
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        loss = get_loss(model, data, target)
        test_loss += loss.data[0]  # sum up batch loss
        if Hp.data_type == 'classification':
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    if Hp.data_type == 'classification':
        print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        print('\n')
