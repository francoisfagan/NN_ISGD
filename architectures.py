""" Defines neural network architectures

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import isgd_fns
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
        return Isgd_RNN(Hp.input_size, Hp.hidden_size, Hp.output_size)
    elif architecture == 'lstm':
        return Isgd_LSTM(Hp.input_size, Hp.hidden_size, Hp.output_size)
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
        self.fc1 = isgd_fns.IsgdArctan(320, 50)  # nn.Linear(320, 50)  #
        self.batch_norm = nn.BatchNorm1d(50, affine=False) if Hp.batch_norm else isgd_fns.IsgdIdentity()
        self.fc2 = isgd_fns.IsgdRelu(50, 10)  # nn.Linear(50, 10)  #

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class Isgd_RNN(nn.Module):
    """ Recurrent neural network architecture
    Based on: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

    To help keep track of dimensions, we use the notation
        h:      Size hidden nodes
        d:      Size of input
        o:      Size of output

    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Isgd_RNN, self).__init__()

        self.hidden_size = hidden_size  # [h]

        self.i2h = isgd_fns.IsgdArctan(input_size + hidden_size, hidden_size)  # [d + h] -> [h]  #nn.Linear
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # [d + h] -> [o]

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden))  # [d + h]

        # Calculate output
        output = self.i2o(combined)  # [o]

        # Calculate next hidden value
        combined = combined.unsqueeze(0)  # [1, d + h]  Introduce a dummy batch size index (required for Isgd* layers)
        hidden = self.i2h(combined)  # [1, h]
        hidden = hidden.squeeze(0)  # [h]  Remove dummy batch size index
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.hidden_size))


class Isgd_LSTM(nn.Module):
    """ LSTM architecture
    Based on: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    and
    http://pytorch.org/docs/master/nn.html

    To help keep track of dimensions, we use the notation
        h:      Size hidden nodes = size of lstm output
        d:      Size of lstm input
        o:      Size of output from layer on top of lstm

    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Isgd_LSTM, self).__init__()

        self.hidden_size = hidden_size  # [h]

        self.lstm = nn.LSTM(input_size, hidden_size)  # [d, h] -> [h, h]
        self.lstm2output = nn.Linear(hidden_size, output_size)  # [h] -> [o]

    def forward(self, input, hidden):
        input = input.unsqueeze(0).unsqueeze(0)  # [d] -> [1 x 1 x d]
        lstm_out, hidden = self.lstm(input, hidden)  # [d, h] -> [h, h]
        output = self.lstm2output(lstm_out)  # [h] -> [o]
        return output, hidden

    def initHidden(self):
        h0 = Variable(torch.zeros(1, 1, self.hidden_size))  # [1 x 1 x h]
        c0 = Variable(torch.zeros(1, 1, self.hidden_size))  # [1 x 1 x h]
        return (h0, c0)  # ([1 x 1 x h], [1 x 1 x h])
