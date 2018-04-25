""" Utility classes and functions are all placed here
"""

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm


class Hp:
    """Stores and sets hyperparameters required for the neural network modules"""

    # None placeholders for hyperparmeters
    batch_norm = None  # True/False indicator of whether to use batch normalization or not
    batch_size = None  # Size of the mini-batches
    clipping_threshold = None  # For gradient clipping. clipping_threshold = 0 means no clipping
    initialization_scale = None  # String that indicates how to initialize the weights and biases
    lr = None  # Learning rate
    mu = None  # L-2 regularization constant
    sgd_type = None  # Whether to use implicit or explicit sgd
    test_batch_size = None  # Size of the batches when evaluating on the test set

    @classmethod
    def get_isgd_hyperparameters(cls):
        return cls.lr, cls.mu, cls.sgd_type

    @classmethod
    def get_initialization_scale(cls, input_features, output_features):
        if cls.initialization_scale == '\sqrt{\frac{6}{n+m}}':
            bias_scale = 0
            weight_scale = math.sqrt(6.0 / (input_features + output_features))
        elif cls.initialization_scale == '0.1':
            bias_scale = 0
            weight_scale = 0.1
        else:
            raise ValueError('initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}')
        return bias_scale, weight_scale


def check_hyperparameters_valid():
    """ Check if hyperparameter values are valid
    """
    assert (
        Hp.batch_size > 1 if Hp.batch_norm else True), 'For nn.BatchNorm1d to work, the batch size has to be greater than 1'
    assert Hp.sgd_type in {'implicit', 'explicit'}, 'sgd_type must be in {implicit, explicit}'
    assert Hp.initialization_scale in {'0.1',
                                       '\sqrt{\frac{6}{n+m}}'}, 'initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}'
    assert Hp.dataset_name in {'mnist', 'addition'}
    assert Hp.architecture in {'conv_ffnn', 'rnn'}

    # Dataset and architecture don't match
    error_string = 'Inappropriate architecture for dataset'
    if Hp.dataset_name == 'mnist':
        assert Hp.architecture in {'conv_ffnn'}, error_string
    elif Hp.dataset_name == 'addition':
        assert Hp.architecture in {'rnn'}, error_string

    # Check if RNN then batch size = 1
    # This is because the sequences can have different lengths
    if Hp.data_type == 'sequential':
        assert Hp.batch_size == 1, 'For RNNs the batch size must be 1. Otherwise the sequences in a batch could have different lengths'


def get_data_type():
    """Get type of data from dataset name stored in Hyperparameters"""
    dataset_name = Hp.dataset_name
    if dataset_name == 'mnist':
        data_type = 'classification'
    elif dataset_name == 'addition':
        data_type = 'sequential'
    else:
        raise ValueError('Data_type not know for given dataset')
    return data_type


def get_optimizer(model):
    """ Get the optimizer from information stored in Hyperparameters

    Args:
        model:  Neural network model

    Returns:
        optimizer:      Optimization algorithm

    """
    if Hp.sgd_type == 'implicit':
        # If implicit then regularization is already done in the backprop,
        # so it shouldn't be included in the optimizer
        return optim.SGD(model.parameters(),
                         lr=Hp.lr,
                         momentum=Hp.momentum)
    else:
        return optim.SGD(model.parameters(),
                         lr=Hp.lr,
                         momentum=Hp.momentum,
                         weight_decay=Hp.mu)


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
        model:          Neural network model
        data [2 x t]:   Mini-batch input data
        target [1]:     Mini-batch target

    Returns:
        loss:           Loss on mini-batch

    """
    # Get rid of zeroth dimension, since the minibatch is of size 1
    data = data[0, :, :]  # [2 x t]

    hidden = model.initHidden()

    sequence_length = data.size()[1]
    for i in range(sequence_length):
        input = data[:, i]
        output, hidden = model(input, hidden)
    return nn.MSELoss()(output, target)


def get_loss(model, data, target):
    """ Return loss for model depending on the type of data,
    which is specified in Hp.data_type

    Args:
        model:  Neural network model
        data:   Mini-batch input data
        target: Mini-batch target

    Returns:
        loss:   Loss on mini-batch

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

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        loss = get_loss(model, data, target)
        loss.backward()

        # Clip gradients
        # As implemented in https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L162-L164
        if Hp.clipping_threshold != 0:
            clip_grad_norm(model.parameters(), Hp.clipping_threshold)

        # Take optimization step
        optimizer.step()

        # Print loss on current datapoint
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.data[0]))


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
        print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    print('')