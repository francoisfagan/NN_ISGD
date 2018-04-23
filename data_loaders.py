""" Defines functions that load data for various datasets
Each function must return a train_loader and test_loader
For more information on train_loaders and test_loaders see
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import Hyperparameters


def mnist():
    """ Loads mnist data as done in
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    Returns:
        train_loader    Loads training data
        test_loader     Loads test data
    """
    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=Hyperparameters.batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=Hyperparameters.test_batch_size, shuffle=True)

    return train_loader, test_loader


def addition_problem(T):
    train_loader = test_loader = None
    return train_loader, test_loader
