"""
Pipeline for testing implicit stochastic gradient descent on neural networks

Author: Francois Fagan, Columbia University
"""

from __future__ import print_function
import json
import architectures
import data_loaders
from utils import Hp, get_optimizer
from train_test import train_and_test

# Set hyperparameters
hyperparameters = {
    'architecture': 'convffnn',  # 'rnn'  # 'lstm'  #
    'batch_size': 100,
    'clipping_threshold': 0.0,
    'dataset_name': 'mnist',  # 'easy_addition'  # 'simple_rnn' #
    'epochs': 3,
    'initialization_scale': '\sqrt{\frac{6}{n+m}}',  # '0.1'
    'lr': 0.1,
    'momentum': 0.0,
    'mu': 0.0,
    'seed': 8,
    'sgdtype': 'explicit'
}

Hp.set_hyperparameters(hyperparameters)
train_loader, test_loader = data_loaders.get_dataset()
model = architectures.get_model()
optimizer = get_optimizer(model)
results = train_and_test(train_loader, test_loader, model, optimizer)

# Save experiment in a dictionary and dump as a json
with open(Hp.get_experiment_name(), 'w') as f:
    experiment_dict = {'hyperparameters': hyperparameters,
                       'results': results}
    json.dump(experiment_dict, f, indent=2)
