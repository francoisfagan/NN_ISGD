""" Defines neural network architectures

"""

import torch.nn as nn
import torch.nn.functional as F
from isgd_fns import IsgdRelu, IsgdIdentity
from utils import Hyperparameters


# Define neural network
class ConvolutionalFFNN(nn.Module):
    def __init__(self):
        super(ConvolutionalFFNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = IsgdRelu(320, 50)  # nn.Linear(320, 50)  #
        self.batch_norm = nn.BatchNorm1d(50, affine=False) if Hyperparameters.batch_norm else IsgdIdentity()
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
