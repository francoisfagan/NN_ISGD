"""
Pipeline for testing implicit stochastic gradient descent on neural networks

Author: Francois Fagan, Columbia University
"""

from __future__ import print_function
import torch
import torch.optim as optim
import architectures
import data_loaders
from utils import train, test, Hyperparameters

# Set seed
torch.manual_seed(1)

# Set hyperparameters
ARCHITECTURE = 'conv_ffnn'
BATCH_NORM = Hyperparameters.batch_norm = False
BATCH_SIZE = Hyperparameters.batch_size = 1
CLIPPING_THRESHOLD = Hyperparameters.clipping_threshold = 0
DATASET = 'mnist'
EPOCHS = Hyperparameters.epochs = 1
INITIALIZATION_SCALE = Hyperparameters.initialization_scale = '\sqrt{\frac{6}{n+m}}'
LEARNING_RATE = Hyperparameters.lr = 0.001
MOMENTUM = Hyperparameters.momentum = 0.0
REGULARIZATION = Hyperparameters.mu = 0.0
SGD_TYPE = Hyperparameters.sgd_type = 'explicit'
TEST_BATCH_SIZE = Hyperparameters.test_batch_size = 64


# Check that hyperparameter settings are valid
assert (BATCH_SIZE > 1 if BATCH_NORM else True), 'For nn.BatchNorm1d to work, the batch size has to be greater than 1'
assert SGD_TYPE in {'implicit', 'explicit'}, 'sgd_type must be in {implicit, explicit}'
assert INITIALIZATION_SCALE in {'0.1',
                                '\sqrt{\frac{6}{n+m}}'}, 'initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}'
assert DATASET in {'mnist', 'addition'}
assert ARCHITECTURE in {'conv_ffnn', 'rnn'}

# Load the data
train_loader, test_loader = data_loaders.mnist()

# Define the model architecture
model = None
if ARCHITECTURE == 'conv_ffnn':
    model = architectures.ConvolutionalFFNN()
elif ARCHITECTURE == 'rnn':
    model = architectures.RNN()

# Define optimizer
if SGD_TYPE == 'implicit':
    # If implicit then regularization is already done in the backprop,
    # so it shouldn't be included in the optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
else:
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=REGULARIZATION)

# Run SGD and test
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)
