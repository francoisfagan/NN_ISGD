""" Utility classes and functions are all placed here
"""

import torch.optim as optim
import math


class Hp:
    """Stores and sets hyperparameters required for the neural network modules"""

    # # None placeholders for hyperparmeters
    architecture = None  # Neural network architecture
    batch_size = None  # Size of the mini-batches
    clipping_threshold = None  # For gradient clipping. clipping_threshold = 0 means no clipping
    dataset_name = None  # Name of the dataset
    epochs = None  # Number of epochs
    initialization_scale = None  # String that indicates how to initialize the weights and biases
    lr = None  # Learning rate
    momentum = None  # Momentum parameter
    mu = None  # Ridge (L-2) regularization constant
    seed = None  # Random seed
    sgd_type = None  # Whether to use implicit or explicit sgd

    # @classmethod
    # def set_hyperparameters(cls):
    #
    #
    #
    #     return cls.lr, cls.mu, cls.sgd_type

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

    assert Hp.sgd_type in {'implicit', 'explicit'}, 'sgd_type must be in {implicit, explicit}'
    assert Hp.initialization_scale in {'0.1',
                                       '\sqrt{\frac{6}{n+m}}'}, 'initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}'
    assert Hp.dataset_name in {'mnist', 'addition', 'easy_addition', 'medium_addition', 'simple_rnn'}
    assert Hp.architecture in {'conv_ffnn', 'rnn', 'lstm'}

    # Dataset and architecture don't match
    error_string = 'Inappropriate architecture for dataset'
    if Hp.dataset_name == 'mnist':
        assert Hp.architecture in {'conv_ffnn'}, error_string
    elif Hp.dataset_name in {'addition', 'simple_rnn'}:
        assert Hp.architecture in {'rnn', 'lstm'}, error_string

    # Check if RNN then batch size = 1
    # This is because the sequences can have different lengths
    if Hp.data_type == 'sequential':
        assert Hp.batch_size == 1, 'For RNNs the batch size must be 1. Otherwise the sequences in a batch could have different lengths'

    # LSTM does not use implicit sgd, so they cannot be combined
    if Hp.architecture == 'lstm':
        assert Hp.sgd_type != 'implicit', 'LSTM does not currently support implicit sgd'


def get_data_type():
    """Get type of data from dataset name stored in Hyperparameters"""
    dataset_name = Hp.dataset_name
    if dataset_name == 'mnist':
        data_type = 'classification'
    elif dataset_name in {'addition', 'easy_addition', 'medium_addition', 'simple_rnn'}:
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
