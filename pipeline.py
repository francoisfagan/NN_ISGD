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

# Hyperparameters
BATCH_NORM = False
BATCH_SIZE = 1
CLIPPING_THRESHOLD = 0
EPOCHS = 1
INITIALIZATION_SCALE = '\sqrt{\frac{6}{n+m}}'
LEARNING_RATE = 0.001
MOMENTUM = 0.0
REGULARIZATION = 0.0
SGD_TYPE = 'implicit'
TEST_BATCH_SIZE = 64

# Check that hyperparameter settings are valid
assert (BATCH_SIZE > 1 if BATCH_NORM else True), 'For nn.BatchNorm1d to work, the batch size has to be greater than 1'
assert SGD_TYPE in {'implicit', 'explicit'}, 'sgd_type must be in {implicit, explicit}'
assert INITIALIZATION_SCALE in {'0.1',
                                '\sqrt{\frac{6}{n+m}}'}, 'initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}'

# Set hyperparameters
Hyperparameters.lr = LEARNING_RATE
Hyperparameters.mu = REGULARIZATION
Hyperparameters.sgd_type = SGD_TYPE
Hyperparameters.initialization_scale = INITIALIZATION_SCALE
Hyperparameters.batch_norm = BATCH_NORM
Hyperparameters.batch_size = BATCH_SIZE
Hyperparameters.test_batch_size = TEST_BATCH_SIZE
Hyperparameters.clipping_threshold = CLIPPING_THRESHOLD

# Load the data
train_loader, test_loader = data_loaders.mnist()

# Define the model architecture
model = architectures.ConvolutionalFFNN()

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
