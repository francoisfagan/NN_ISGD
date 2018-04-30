"""
Pipeline for testing implicit stochastic gradient descent on neural networks

Author: Francois Fagan, Columbia University
"""

from __future__ import print_function
import time
import torch
import architectures
import data_loaders
from utils import Hp, get_data_type, get_optimizer, check_hyperparameters_valid
from train_test import train, test

# Set seed
torch.manual_seed(1)

# Set hyperparameters
Hp.architecture = 'conv_ffnn'  # 'rnn'  # 'lstm'  #
Hp.batch_norm = False
Hp.batch_size = 1
Hp.clipping_threshold = 0.0
Hp.dataset_name = 'mnist'  # 'easy_addition'  # 'simple_rnn' #
Hp.epochs = 1
Hp.initialization_scale = '\sqrt{\frac{6}{n+m}}'  # '0.1'  #
Hp.lr = 0.5
Hp.momentum = 0.0
Hp.mu = 0.0  # 1e-4
Hp.sgd_type = 'explicit'

# Hyperpameters for RNN
Hp.train_length = 10000
Hp.test_length = 300
Hp.sequence_length = 11
Hp.input_size = 2
Hp.hidden_size = 10
Hp.output_size = 1

# Infer the data type from the dataset
Hp.data_type = get_data_type()

# Check that all of the hyperparameters are valid
check_hyperparameters_valid()

# Using the hyperparameters specified above, load the data, model and optimizer
train_loader, test_loader = data_loaders.get_dataset()

model = architectures.get_model()
optimizer = get_optimizer(model)

# Traing and test
time_start = time.time()
print('Started training')
for epoch in range(Hp.epochs):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
time_finish = time.time()

# Print how long it took to run the algorithm
time_total = time_finish - time_start
print('Time: ', time_total)
