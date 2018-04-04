from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ISGD_fns import *

# Set seed
torch.manual_seed(1)

# Hyperparameters
BATCH_SIZE = 1
TEST_BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.001
REGULARIZATION = 0.0

# Update learning rate for ISGD
IsgdUpdate.set_lr(LEARNING_RATE)
IsgdUpdate.set_regularization(REGULARIZATION)

# Load the data
train_loader = DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True)


# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # ESGD original
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # ISGD original
        self.fc1 = IsgdLinear(320, 50)
        self.fc2 = IsgdLinear(50, 10)

        # ISGD simple
        # self.fc_relu = IsgdRelu(784, 10)
        # self.fc_linear = IsgdLinear(10, 10)

    def forward(self, x):

        # # ESGD original
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        # ISGD original
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # # ISGD simple
        # x = x.view(-1, 784)
        # x = self.fc_relu(x)
        # x = self.fc_linear(x)

        return F.log_softmax(x, dim=1)


# Define optimizer
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)


# Define training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


# Define testing
def test():
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


# Run SGD and test
for epoch in range(EPOCHS):
    train(epoch)
    test()
