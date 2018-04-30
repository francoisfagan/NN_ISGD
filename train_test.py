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

    # Cumulative performance measures and counts
    cum_loss = 0  # Loss
    cum_minibatches = 0  # Number of minibatches

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        loss = get_loss(model, data, target)
        loss.backward()

        # Update performance measures and counts
        cum_loss += loss.data[0]
        cum_minibatches += 1

        # Clip gradients
        # As implemented in https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L162-L164
        if Hp.clipping_threshold != 0:
            clip_grad_norm(model.parameters(), Hp.clipping_threshold)

        # Take optimization step
        optimizer.step()

        # Print average loss
        average_loss = cum_loss / cum_minibatches
        if batch_idx % 100 == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), average_loss))

            # Reset cumulative performance measures and counts
            cum_loss = 0
            cum_minibatches = 0


# Define testing
def test(model, loader, dataset):
    """ Tests the neural network model

    Args:
        model:          Neural network model
        loader:    Class that loads mini-batches
        dataset:        'train' or 'test'

    Returns:
        prints average loss

    """
    model.eval()

    # Cumulative performance measures and counts
    cum_loss = 0  # Loss
    cum_correct = 0  # Fraction of correct predictions if classification
    cum_minibatches = 0  # Number of minibatches
    cum_datapoints = 0  # Number of datapoints

    for data, target in loader:
        data, target = Variable(data, volatile=True), Variable(target)
        loss = get_loss(model, data, target)

        # Update performance measures and counts
        cum_loss += loss.data[0]  # loss.data[0] is a scalar equal to the mean loss over the minibatch
        cum_minibatches += 1
        cum_datapoints += data.size()[0]
        if Hp.data_type == 'classification':
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            cum_correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    # Record and print performance measures
    average_loss = cum_loss / cum_minibatches
    print('{} set: average loss: {:.4f}'.format(dataset, average_loss))

    average_accuracy = 0
    if Hp.data_type == 'classification':
        average_accuracy = 100. * cum_correct / cum_datapoints
        print('{} set: accuracy: {}/{} ({:.0f}%)'.format(
            dataset, cum_correct, cum_datapoints, average_accuracy))

    epoch_performance = {'average_loss': average_loss, 'average_accuracy': average_accuracy}

    return epoch_performance
