"""
Pipeline for testing implicit stochastic gradient descent on neural networks

Author: Francois Fagan, Columbia University
"""

from __future__ import print_function
import json
import architectures
import data_loaders
from utils import Hp, get_optimizer, get_hyperparameters
from train_test import train_and_test

# Load all hyperparameter permutations
hyperparameter_list_name = 'mnist_experiments'
for hyperparameters in get_hyperparameters(hyperparameter_list_name):
    # Run experiment for each hyperparameter
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
