"""
Pipeline for testing implicit stochastic gradient descent on neural networks

Author: Francois Fagan, Columbia University
"""

from __future__ import print_function
import torch
import architectures
import data_loaders
from utils import Hp, get_data_type, get_optimizer, check_hyperparameters_valid
from train_test import train, test

# Set seed
torch.manual_seed(1)

# Set hyperparameters
Hp.architecture = 'lstm' #'rnn'#'conv_ffnn'  #
Hp.batch_norm = False
Hp.batch_size = 1
Hp.clipping_threshold = 0
Hp.dataset_name = 'simple_rnn' #'addition' #'mnist'  #
Hp.epochs = 1
Hp.initialization_scale = '\sqrt{\frac{6}{n+m}}'
Hp.lr = 0.01
Hp.momentum = 0.0
Hp.mu = 0.0
Hp.sgd_type = 'explicit'
Hp.test_batch_size = 64

# Hyperpameters for RNN
Hp.train_length = 20000
Hp.test_length = 3000
Hp.sequence_length = 11
Hp.input_size = 2
Hp.hidden_size = 1
Hp.output_size = 1

# Infer the data type from the dataset
Hp.data_type = get_data_type()

# Check that all of the hyperparameters are valid
check_hyperparameters_valid()

# Using the hyperparameters specified above, load the data, model and optimizer
train_loader, test_loader = data_loaders.get_dataset()

model = architectures.get_model()
optimizer = get_optimizer(model)

# Run SGD and test
for epoch in range(Hp.epochs):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
