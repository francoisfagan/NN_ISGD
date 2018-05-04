""" Code for calculating losses, training and testing neural network models

"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from utils import Hp
import time


def train_and_test(train_loader, test_loader, model, optimizer):
    """ Train and test model using dataloaders with the given optimizer

    Args:
        train_loader:   Loader of training dataset
        test_loader:    Loader of testing dataset
        model:          Neural network model
        optimizer:      Neural network optimizer

    Returns:
        results:        Dictionary storing the average loss and accuracy per epoch
                            on both the training and testing sets

    """
    # Initialize variables to store results
    time_start = time.time()
    results = {'train': {'average_loss': [],
                         'average_accuracy': []},
               'test': {'average_loss': [],
                        'average_accuracy': []}}

    print('Started training')
    for epoch in range(Hp.hp['epochs']):
        # Train for one epoch
        average_loss_intra_epoch, average_accuracy_intra_epoch = train(model, train_loader, optimizer, epoch)

        # Record intra-epoch loss (e.g. every 100 minibatches)
        results['average_loss_intra_epoch'] = average_loss_intra_epoch
        results['average_accuracy_intra_epoch'] = average_accuracy_intra_epoch

        # Record training and test loss
        for dataset, loader in [('test', test_loader), ('train', train_loader)]:
            average_loss, average_accuracy = test(model, loader, dataset)
            results[dataset]['average_loss'].append(average_loss)
            results[dataset]['average_accuracy'].append(average_accuracy)

        # Add intra-epoch loss for the end of the epoch
        results['average_loss_intra_epoch'].append(average_loss)
        results['average_accuracy_intra_epoch'].append(average_accuracy)

    time_finish = time.time()

    # Print how long it took to run the algorithm
    results['runtime'] = time_finish - time_start

    return results


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


def music_loss(model, data, target):
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
    target = target[0, :, :]  # [d x t]

    hidden = model.initHidden()

    sequence_length = data.size()[0]
    loss = 0
    for i in range(sequence_length):
        input = data[i, :]  # [d]
        output, hidden = model(input, hidden)

        # Normalize the output to be between 0 and 1
        # since it needs to be a probability as doing prediction
        output = (output + np.pi / 2) / np.pi

        # Calculate the log-loss
        loss += -(target[i, :] * torch.log(output)
                  + (1 - target[i, :]) * (torch.log(1 - output))).mean()

    return loss


def autoencoder_loss(model, data):
    """ Return loss for classification models

    Args:
        model:  Neural network model
        data:   Mini-batch input data

    Returns:
        loss:   Loss on mini-batch

    """
    output = model(data)
    return nn.MSELoss()(output, data) / 1000.0


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
    if Hp.hp['data_type'] == 'classification':
        return classification_loss(model, data, target)
    elif Hp.hp['data_type'] == 'sequential':
        return rnn_loss(model, data, target)
    elif Hp.hp['data_type'] == 'sequential_many':
        return music_loss(model, data, target)
    elif Hp.hp['data_type'] == 'autoencoder':
        return autoencoder_loss(model, data)
    else:
        raise ValueError('No valid loss function for')


# Define training
def train(model, train_loader, optimizer, epoch):
    """ Trains the neural network model

    Args:
        model:          Neural network model
        train_loader:   Class that loads mini-batches
        optimizer:      Optimization algorithm
        epoch:          Current epoch number

    """
    print('')
    model.train()

    # Cumulative performance measures and counts
    cum_loss = 0  # Loss
    cum_minibatches = 0  # Number of minibatches

    average_loss_intra_epochs = []
    average_accuracy_intra_epochs = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data and put it on gpu if gpu available
        data, target = Variable(data).to(Hp.device), Variable(target).to(Hp.device)

        if Hp.hp['inner_ISGD_iterations'] == 0:
            # Take SGD step
            optimizer.zero_grad()
            loss = get_loss(model, data, target)
            loss.backward()

            # Clip gradients
            # As implemented in https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L162-L164
            if Hp.hp['clipping_threshold'] != 0:
                clip_grad_norm_(model.parameters(), Hp.hp['clipping_threshold'])

            # Take optimization step
            optimizer.step()
        # elif batch_idx % Hp.hp['inner_ISGD_iterations'] == 0:
        else:
            original_param = dict()
            for j, param in enumerate(model.parameters()):
                original_param[j] = param.clone()
            for i in range(Hp.hp['inner_ISGD_iterations']):
                loss = get_loss(model, data, target)
                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for j, param in enumerate(model.parameters()):
                        # param -= (Hp.hp['lr'] * param.grad + param - original_param[j]
                        #           * 2.0 / float(Hp.hp['inner_ISGD_iterations']))
                        param -= (Hp.hp['lr'] * (param.grad + param - original_param[j])
                                  / np.sqrt(float(Hp.hp['inner_ISGD_iterations'])))

        # Update performance measures and counts
        cum_loss += loss.data
        cum_minibatches += 1

        # Print average loss
        average_loss = cum_loss / cum_minibatches
        if batch_idx % 10 == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), average_loss))

            if 'intra_epoch' in Hp.hp and Hp.hp['intra_epoch']:
                loss_intra_epoch, accuracy_intra_epoch = test(model, train_loader, 'train')
                average_loss_intra_epochs.append(loss_intra_epoch)
                average_accuracy_intra_epochs.append(accuracy_intra_epoch)

                model.train()

            # Reset cumulative performance measures and counts
            cum_loss = 0
            cum_minibatches = 0

    print('')
    return average_loss_intra_epochs, average_accuracy_intra_epochs


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
        # Get data and put it on gpu if gpu available
        data, target = Variable(data).to(Hp.device), Variable(target).to(Hp.device)
        loss = get_loss(model, data, target)

        # Update performance measures and counts
        cum_loss += float(loss.data)  # loss.data is a scalar equal to the mean loss over the minibatch
        cum_minibatches += 1
        cum_datapoints += data.size()[0]
        if Hp.hp['data_type'] == 'classification':
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            cum_correct += int(pred.eq(target.data.view_as(pred)).long().cpu().sum())

    # Record and print performance measures
    average_loss = cum_loss / cum_minibatches
    print('{} set: average loss: {:.4f}'.format(dataset, average_loss))

    average_accuracy = 0
    if Hp.hp['data_type'] == 'classification':
        average_accuracy = 100. * cum_correct / cum_datapoints
        print('{} set: accuracy: {}/{} ({:.0f}%)'.format(
            dataset, cum_correct, cum_datapoints, average_accuracy))

    return average_loss, average_accuracy
