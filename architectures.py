""" Defines neural network architectures

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from isgd_fns import IsgdRelu, IsgdIdentity
from utils import Hp


def get_model():
    """ Get model with the architecture stored in Hyperparameters

    Returns:
        Neural network model

    """
    architecture = Hp.architecture
    if architecture == 'conv_ffnn':
        return ConvolutionalFFNN()
    elif architecture == 'rnn':
        return RNN(Hp.input_size, Hp.hidden_size, Hp.output_size)
    else:
        raise ValueError('There is no model for the given architecture')


# Define neural network
class ConvolutionalFFNN(nn.Module):
    """ Convolutional Feed Forward Neural Network architecture
    Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py

    """

    def __init__(self):
        super(ConvolutionalFFNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = IsgdRelu(320, 50)  # nn.Linear(320, 50)  #
        self.batch_norm = nn.BatchNorm1d(50, affine=False) if Hp.batch_norm else IsgdIdentity()
        self.fc2 = IsgdRelu(50, 10)  # nn.Linear(50, 10)  #

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class RNN(nn.Module):
    """ Recurrent neural network architecture
    Based on: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden))
        hidden = self.i2h(combined)
        # hidden = nn.functional.sigmoid(hidden)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.hidden_size))
