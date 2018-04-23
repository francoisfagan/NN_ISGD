""" Utility classes and functions are all placed here
"""

import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm


class Hyperparameters:
    """Stores and sets hyperparameters required for the neural network modules"""

    # None placeholders for hyperparmeters
    batch_norm = None               # True/False indicator of whether to use batch normalization or not
    batch_size = None               # Size of the mini-batches
    clipping_threshold = None       # For gradient clipping. clipping_threshold = 0 means no clipping
    initialization_scale = None     # String that indicates how to initialize the weights and biases
    lr = None                       # Learning rate
    mu = None                       # L-2 regularization constant
    sgd_type = None                 # Whether to use implicit or explicit sgd
    test_batch_size = None          # Size of the batches when evaluating on the test set

    @classmethod
    def get_isgd_hyperparameters(cls):
        return cls.lr, cls.mu, cls.sgd_type

    @classmethod
    def get_initialization_scale(cls, input_features, output_features):
        if cls.initialization_scale == '\sqrt{\frac{6}{n+m}}':
            bias_scale = 0
            weight_scale = math.sqrt(6.0 / (input_features + output_features))
        elif cls.initialization_scale == '0.1':
            bias_scale = 0
            weight_scale = 0.1
        else:
            raise ValueError('initialization_scale must be in {0.1, \sqrt{\frac{6}{n+m}}}')
        return bias_scale, weight_scale


# Define training
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # Clip gradients
        # As implemented in https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L162-L164
        if Hyperparameters.clipping_threshold != 0:
            clip_grad_norm(model.parameters(), Hyperparameters.clipping_threshold)

        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.data[0]))


# Define testing
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
