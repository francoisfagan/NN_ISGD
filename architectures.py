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
    architecture = Hp.hp['architecture']
    model = None
    if architecture == 'convffnn':
        model = ConvolutionalFFNN()
    elif architecture == 'rnn':
        model = Isgd_RNN(Hp.hp['input_size'], Hp.hp['hidden_size'], Hp.hp['output_size'])
    elif architecture == 'music':
        model = Isgd_RNN(88, Hp.hp['nodes'], 88)
    elif architecture == 'lstm':
        model = Isgd_LSTM(Hp.hp['input_size'], Hp.hp['hidden_size'], Hp.hp['output_size'])
    elif architecture == 'autoencoder':
        model = Autoencoder()
    elif architecture == 'classification':
        model = Classification()
    else:
        raise ValueError('There is no model for the given architecture')

    # If have a gpu then put the model on the gpu
    model = model.to(Hp.device)

    return model


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
        self.fc2 = isgd_fns.IsgdRelu(50, 10)  # nn.Linear(50, 10)  #

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class Classification(nn.Module):
    """
    Autoencoder architecture as specified in
    'Training Neural Networks with Stochastic Hessian-Free Optimization'

    It only uses sigmoidal activations except for the first and final layer with is a relu.
    It has the following structure:
    784-500-250-30

    """

    def __init__(self):
        super(Classification, self).__init__()

        # Get the number of classes and input dimension
        classes = Hp.classes
        input_dim = Hp.input_dim

        # Layers
        self.f1 = isgd_fns.IsgdArctan(input_dim, input_dim)
        self.f2 = isgd_fns.IsgdArctan(input_dim, input_dim)
        self.f3 = isgd_fns.IsgdArctan(input_dim, input_dim)
        self.ffinal = nn.Linear(input_dim, classes)

    def forward(self, x):

        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        # x = self.f4(x)
        # x = self.f5(x)
        x = self.ffinal(x)

        return F.log_softmax(x, dim=1)


class Autoencoder(nn.Module):
    """
    Autoencoder architecture as specified in
    'Training Neural Networks with Stochastic Hessian-Free Optimization'

    It only uses sigmoidal activations except for the first and final layer with is a relu.
    It has the following structure:
    784-500-250-30

    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        # self.f = nn.Linear(784, 784)

        self.f1 = isgd_fns.IsgdRelu(784, 500)  # nn.Linear(784, 500) #
        self.f2 = isgd_fns.IsgdRelu(500, 300)
        self.f3 = isgd_fns.IsgdRelu(300, 100)
        self.f4 = isgd_fns.IsgdRelu(100, 30)
        self.f5 = isgd_fns.IsgdRelu(30, 100)
        self.f6 = isgd_fns.IsgdRelu(100, 300)
        self.f7 = isgd_fns.IsgdRelu(300, 500)
        self.f8 = isgd_fns.IsgdRelu(500, 784)  # nn.Linear(500, 784) #

    def forward(self, x):
        # x = self.f(x)

        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)

        return x


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
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)  # [d + h] -> [o]
        self.i2o = isgd_fns.IsgdArctan(input_size + hidden_size, output_size)  # [d + h] -> [o]

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden))  # [d + h]

        # Introduce a dummy batch size index (required for Isgd* layers)
        combined = combined.unsqueeze(0)  # [1, d + h]

        # Calculate output
        output = self.i2o(combined)  # [o]

        # Calculate next hidden value
        hidden = self.i2h(combined)  # [1, h]
        hidden = hidden.squeeze(0)  # [h]  Remove dummy batch size index
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.hidden_size)).to(Hp.device)


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
