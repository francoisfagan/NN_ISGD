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

# Set hyperparameters
Hp.architecture = 'conv_ffnn'  # 'rnn'  # 'lstm'  #
Hp.batch_size = 100
Hp.clipping_threshold = 0.0
Hp.dataset_name = 'mnist'  # 'easy_addition'  # 'simple_rnn' #
Hp.epochs = 2
Hp.initialization_scale = '\sqrt{\frac{6}{n+m}}'  # '0.1'  #
Hp.lr = 0.1
Hp.momentum = 0.0
Hp.mu = 0.0  # 1e-4
Hp.seed = 10
Hp.sgd_type = 'explicit'

# Set seed
torch.manual_seed(Hp.seed)

# Infer the data type from the dataset
Hp.data_type = get_data_type()

# Check that all of the hyperparameters are valid
check_hyperparameters_valid()

# Using the hyperparameters specified above, load the data, model and optimizer
train_loader, test_loader = data_loaders.get_dataset()
model = architectures.get_model()
optimizer = get_optimizer(model)

# Train and test
time_start = time.time()
performance = {'train': [], 'test': []}
print('Started training')
for epoch in range(Hp.epochs):
    train(model, train_loader, optimizer, epoch)
    print('')
    for dataset, loader in [('train', train_loader), ('test', test_loader)]:
        performance[dataset].append(test(model, loader, dataset))
    print('\n')
time_finish = time.time()

# Print how long it took to run the algorithm
time_total = time_finish - time_start
print('Time: ', time_total)

print(performance)
