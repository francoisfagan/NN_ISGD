""" Defines functions that load data for various datasets
Each function must return a train_loader and test_loader
For more information on train_loaders and test_loaders see
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import Hp


class AdditionDataset(Dataset):
    """Addition dataset as introduced in the original LSTM paper.
    This implementation is from p.11 of 'On the difficulty of training recurrent neural networks' """

    def __init__(self, dataset_length, len_sequence):
        self.dataset_length = dataset_length  # This is what is returned by len(), see def __len__(self) below
        self.t = len_sequence  # Length of sequence
        # Check that sequence length is at least 10
        # If not, there is no randomness in the position of the first number to be added
        assert (self.t > 10), 'Sequence length must be at least 10'

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, dummy_index):
        # The dummy index is required for the dataloader to work,
        # but since we are sampling data randomly it has no effect

        # Sample the length of the sequence and positions of numbers to add
        t_dash = np.random.randint(self.t, int(self.t * 11.0 / 10.0))  # Length of the sequence
        t_1 = np.random.randint(0, int(t_dash / 10.0))  # Indicator of position of first number to add
        t_2 = np.random.randint(int(t_dash / 10.0), int(t_dash / 2.0))  # Indicator of position of second number to add

        # We generate random numbers uniformly sampled from [0,1]
        # as depicted in Figure 2 of
        # "Learning Recurrent Neural Networks with Hessian-Free Optimization"
        # Details of how to sample the numbers was not given in
        # "On the difficulty of training recurrent neural networks"
        sequence = torch.zeros((2, t_dash))  # Initialize empty sequence
        sequence[0, :] = torch.rand((1, t_dash))  # Make first row random numbers

        # Set second row to indicate which numbers to add
        sequence[1, t_1] = 1.0
        sequence[1, t_2] = 1.0

        # Calculate target
        target = torch.Tensor([sequence[0, t_1] + sequence[0, t_2]])

        # Collect sequence and target into a sample
        sample = (sequence, target)

        return sample

class EasyAdditionDataset(Dataset):
    """Easier version of the addition dataset
    Instead of the numbers to be summed being at the beginning and middle,
    they are next to each other and towards the end of the sequence
    """

    def __init__(self, dataset_length, len_sequence):
        self.dataset_length = dataset_length  # This is what is returned by len(), see def __len__(self) below
        self.t = len_sequence  # Length of sequence
        # Check that sequence length is at least 5 so that there is sufficient randomness
        assert (self.t > 5), 'Sequence length must be at least 5'

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, dummy_index):
        # The dummy index is required for the dataloader to work,
        # but since we are sampling data randomly it has no effect

        # Sample the length of the sequence and positions of numbers to add
        t_dash = np.random.randint(self.t, int(self.t * 11.0 / 10.0))  # Length of the sequence
        t_2 = np.random.randint(int(t_dash * 9 / 10.0), t_dash)  # Indicator of position of second number to add
        t_1 = t_2 - 1  # Indicator of position of first number to add

        # We generate random numbers uniformly sampled from [0,1]
        # as depicted in Figure 2 of
        # "Learning Recurrent Neural Networks with Hessian-Free Optimization"
        # Details of how to sample the numbers was not given in
        # "On the difficulty of training recurrent neural networks"
        sequence = torch.zeros((2, t_dash))  # Initialize empty sequence
        sequence[0, :] = torch.rand((1, t_dash))  # Make first row random numbers

        # Set second row to indicate which numbers to add
        sequence[1, t_1] = 1.0
        sequence[1, t_2] = 1.0

        # Calculate target
        target = torch.Tensor([sequence[0, t_1] + sequence[0, t_2]])

        # Collect sequence and target into a sample
        sample = (sequence, target)

        return sample

class MediumAdditionDataset(Dataset):
    """Easier version of the addition dataset
    Instead of the numbers to be summed being at the beginning and middle,
    they are both at the middle and the end
    """

    def __init__(self, dataset_length, len_sequence):
        self.dataset_length = dataset_length  # This is what is returned by len(), see def __len__(self) below
        self.t = len_sequence  # Length of sequence
        # Check that sequence length is at least 10
        # If not, there is no randomness in the position of the first number to be added
        assert (self.t > 10), 'Sequence length must be at least 10'

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, dummy_index):
        # The dummy index is required for the dataloader to work,
        # but since we are sampling data randomly it has no effect

        # Sample the length of the sequence and positions of numbers to add
        t_dash = np.random.randint(self.t, int(self.t * 11.0 / 10.0))  # Length of the sequence
        t_1 = t_dash - 1 - np.random.randint(0, int(t_dash / 10.0))  # Indicator of position of first number to add
        t_2 = t_dash - 1 - np.random.randint(int(t_dash / 10.0), int(t_dash / 2.0))  # Indicator of position of second number to add

        # We generate random numbers uniformly sampled from [0,1]
        # as depicted in Figure 2 of
        # "Learning Recurrent Neural Networks with Hessian-Free Optimization"
        # Details of how to sample the numbers was not given in
        # "On the difficulty of training recurrent neural networks"
        sequence = torch.zeros((2, t_dash))  # Initialize empty sequence
        sequence[0, :] = torch.rand((1, t_dash))  # Make first row random numbers

        # Set second row to indicate which numbers to add
        sequence[1, t_1] = 1.0
        sequence[1, t_2] = 1.0

        # Calculate target
        target = torch.Tensor([sequence[0, t_1] + sequence[0, t_2]])

        # Collect sequence and target into a sample
        sample = (sequence, target)

        return sample


class SimpleRNN(Dataset):
    """Addition dataset as introduced in the original LSTM paper.
    This implementation is from p.11 of 'On the difficulty of training recurrent neural networks' """

    def __init__(self, dataset_length, len_sequence):
        self.dataset_length = dataset_length  # This is what is returned by len(), see def __len__(self) below
        self.t = len_sequence  # Length of sequence
        # Check that sequence length is at least 10
        # If not, there is no randomness in the position of the first number to be added
        assert (self.t > 10), 'Sequence length must be at least 10'

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, dummy_index):
        # The dummy index is required for the dataloader to work,
        # but since we are sampling data randomly it has no effect

        # Sample the length of the sequence and positions of numbers to add
        t_dash = np.random.randint(self.t, int(self.t * 11.0 / 10.0))  # Length of the sequence

        # Generate random sequence
        sequence = torch.rand((2, t_dash))  # Initialize empty sequence

        # Set target to be sum of final values
        target = torch.Tensor([torch.sum(sequence[:, -1])])

        # Collect sequence and target into a sample
        sample = (sequence, target)

        return sample


def get_dataset():
    """ Return the train and test loaders for the dataset

    Args:
        dataset_name: Name of datast

    Returns:
        train_loader    Loads training data
        test_loader     Loads test data

    """
    print('Loading dataset')
    dataset_name = Hp.dataset_name
    if dataset_name == 'mnist':
        return mnist()
    elif dataset_name == 'addition':
        return addition_problem(Hp.train_length,
                                Hp.test_length,
                                Hp.sequence_length
                                )
    elif dataset_name == 'easy_addition':
        return easy_addition_problem(Hp.train_length,
                                Hp.test_length,
                                Hp.sequence_length
                                )
    elif dataset_name == 'medium_addition':
        return medium_addition_problem(Hp.train_length,
                                Hp.test_length,
                                Hp.sequence_length
                                )
    elif dataset_name == 'simple_rnn':
        return simple_rnn_problem(Hp.train_length,
                                Hp.test_length,
                                Hp.sequence_length
                                )


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
        batch_size=Hp.batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=Hp.test_batch_size, shuffle=True)

    return train_loader, test_loader


def addition_problem(train_length, test_length, sequence_length, num_workers=4):
    """
    This is the addition problem


    Args:
        train_length:       Number of training examples for each epoch
        test_length:        Number of test examples for each test
        sequence_length:    Length of each sequence
        num_workers:        Number of workers loading the data

    Returns:
        train_loader    Loads training data
        test_loader     Loads test data

    """
    # Batch size should be 1 to prevent sequences in the same batch having different lengths
    batch_size = 1

    train_loader = DataLoader(AdditionDataset(train_length, sequence_length),
                              batch_size=batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(AdditionDataset(test_length, sequence_length),
                             batch_size=batch_size,
                             num_workers=num_workers)
    return train_loader, test_loader


def easy_addition_problem(train_length, test_length, sequence_length, num_workers=4):
    """
    This is the addition problem


    Args:
        train_length:       Number of training examples for each epoch
        test_length:        Number of test examples for each test
        sequence_length:    Length of each sequence
        num_workers:        Number of workers loading the data

    Returns:
        train_loader    Loads training data
        test_loader     Loads test data

    """
    # Batch size should be 1 to prevent sequences in the same batch having different lengths
    batch_size = 1

    train_loader = DataLoader(EasyAdditionDataset(train_length, sequence_length),
                              batch_size=batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(EasyAdditionDataset(test_length, sequence_length),
                             batch_size=batch_size,
                             num_workers=num_workers)
    return train_loader, test_loader


def medium_addition_problem(train_length, test_length, sequence_length, num_workers=4):
    """
    This is the addition problem


    Args:
        train_length:       Number of training examples for each epoch
        test_length:        Number of test examples for each test
        sequence_length:    Length of each sequence
        num_workers:        Number of workers loading the data

    Returns:
        train_loader    Loads training data
        test_loader     Loads test data

    """
    # Batch size should be 1 to prevent sequences in the same batch having different lengths
    batch_size = 1

    train_loader = DataLoader(MediumAdditionDataset(train_length, sequence_length),
                              batch_size=batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(MediumAdditionDataset(test_length, sequence_length),
                             batch_size=batch_size,
                             num_workers=num_workers)
    return train_loader, test_loader


def simple_rnn_problem(train_length, test_length, sequence_length, num_workers=4):
    """
    This is a simple RNN problem where the target is the sum of the final values in the sequence
    This is very easy and can be used as a check of whether the pipeline is working


    Args:
        train_length:       Number of training examples for each epoch
        test_length:        Number of test examples for each test
        sequence_length:    Length of each sequence
        num_workers:        Number of workers loading the data

    Returns:
        train_loader    Loads training data
        test_loader     Loads test data

    """
    # Batch size should be 1 to prevent sequences in the same batch having different lengths
    batch_size = 1

    train_loader = DataLoader(SimpleRNN(train_length, sequence_length),
                              batch_size=batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(SimpleRNN(test_length, sequence_length),
                             batch_size=batch_size,
                             num_workers=num_workers)
    return train_loader, test_loader
