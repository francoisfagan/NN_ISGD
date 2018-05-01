""" Utility classes and functions are all placed here
"""

import torch.optim as optim
import torch
import math
import itertools
import json


class Hp:
    """Stores and sets hyperparameters required for the neural network modules as a dictionary
    The hyperparameters are:

        architecture            # Neural network architecture
        batch_size              # Size of the mini-batches
        clipping_threshold      # For gradient clipping. clipping_threshold = 0 means no clipping
        dataset_name            # Name of the dataset
        epochs                  # Number of epochs
        initialization_scale    # String that indicates how to initialize the weights and biases
        lr                      # Learning rate
        momentum                # Momentum parameter
        mu                      # Ridge (L-2) regularization constant
        seed                    # Random seed
        sgd_type                # Whether to use implicit or explicit sgd

    """

    # Initialize hyperparameter dictionary
    hp = None

    @classmethod
    def set_hyperparameters(cls, hyperparameters):
        """ Store all of the hyperparameters in the Hp class

        Args:
            hyperparameters:    Dictionary of hyperparameters

        """
        print(json.dumps(hyperparameters, indent=2))

        # Store the hyperparameters
        cls.hp = hyperparameters

        # Set seed
        torch.manual_seed(cls.hp['seed'])

        # Infer the data type from the dataset
        cls.hp['data_type'] = cls.get_data_type()

        # Check that all of the hyperparameters are valid
        cls.check_hyperparameters_valid()

        # # Determine if can run on GPU or CPU
        # cls.gpu = torch.cuda.is_available()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def check_hyperparameters_valid(cls):
        """ Check if hyperparameter values are valid
        """

        assert cls.hp['sgdtype'] in {'implicit', 'explicit'}, 'sgd_type must be in {implicit, explicit}'
        assert cls.hp['initialization_scale'] in {'0.1',
                                                  '\sqrt{\frac{6}{n+m}}'}, 'initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}'
        assert cls.hp['dataset_name'] in {'mnist', 'addition', 'easy_addition', 'medium_addition', 'simple_rnn'}
        assert cls.hp['architecture'] in {'convffnn', 'rnn', 'lstm'}

        # Dataset and architecture don't match
        error_string = 'Inappropriate architecture for dataset'
        if cls.hp['dataset_name'] == 'mnist':
            assert cls.hp['architecture'] in {'convffnn'}, error_string
        elif cls.hp['dataset_name'] in {'addition', 'simple_rnn'}:
            assert cls.hp['architecture'] in {'rnn', 'lstm'}, error_string

        # Check if RNN then batch size = 1
        # This is because the sequences can have different lengths
        if cls.hp['data_type'] == 'sequential':
            assert cls.hp[
                       'batch_size'] == 1, 'For RNNs the batch size must be 1. Otherwise the sequences in a batch could have different lengths'

        # LSTM does not use implicit sgd, so they cannot be combined
        if cls.hp['architecture'] == 'lstm':
            assert cls.hp['sgd_type'] != 'implicit', 'LSTM does not currently support implicit sgd'

    @classmethod
    def get_data_type(cls):
        """Get type of data from dataset name stored in Hyperparameters"""
        dataset_name = cls.hp['dataset_name']
        if dataset_name == 'mnist':
            data_type = 'classification'
        elif dataset_name in {'addition', 'easy_addition', 'medium_addition', 'simple_rnn'}:
            data_type = 'sequential'
        else:
            raise ValueError('Data_type not know for given dataset')
        return data_type

    @classmethod
    def get_isgd_hyperparameters(cls):
        return cls.hp['lr'], cls.hp['mu'], cls.hp['sgdtype']

    @classmethod
    def get_initialization_scale(cls, input_features, output_features):
        initialization_scale = cls.hyperparameters['initialization_scale']
        if initialization_scale == '\sqrt{\frac{6}{n+m}}':
            bias_scale = 0
            weight_scale = math.sqrt(6.0 / (input_features + output_features))
        elif initialization_scale == '0.1':
            bias_scale = 0
            weight_scale = 0.1
        else:
            raise ValueError('initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}')
        return bias_scale, weight_scale

    @classmethod
    def get_experiment_name(cls):
        """ Create and return the file name of the experiment,
        summarizing the hyperparameter values in one string

        Returns:    File name of experiment

        """
        return 'results/' + '|'.join(key[:4] + '_' + str(value)[:6] for key, value in cls.hp.items()) + '.json'


def get_hyperparameters(hyperparameter_list_name):
    """ Yield all permutations of hyperparameters given their potential values

    Args:
        hyperparameter_list_name: name of json file in which the hyperparameter values are stored

    Returns:
        Generatore of hyperparameter values

    """
    # Open json file containing the hyperparameter lists of potential values
    with open('hyperparameter_lists/' + hyperparameter_list_name + '.json') as f:
        hyperparameter_lists = dict(json.load(f))

    # Yield all permutations of values
    list_of_hyperparameter_tuples = [[(key, value) for value in values] for key, values in hyperparameter_lists.items()]
    for hyperparameter_tuple in itertools.product(*list_of_hyperparameter_tuples):
        hyperparameter = dict(hyperparameter_tuple)
        yield hyperparameter


def get_optimizer(model):
    """ Get the optimizer from information stored in Hyperparameters

    Args:
        model:  Neural network model

    Returns:
        optimizer:      Optimization algorithm

    """
    if Hp.hp['sgdtype'] == 'implicit':
        # If implicit then regularization is already done in the backprop,
        # so it shouldn't be included in the optimizer
        return optim.SGD(model.parameters(),
                         lr=Hp.hp['lr'],
                         momentum=Hp.hp['momentum'])
    else:
        return optim.SGD(model.parameters(),
                         lr=Hp.hp['lr'],
                         momentum=Hp.hp['momentum'],
                         weight_decay=Hp.hp['mu'])
