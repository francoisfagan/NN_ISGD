from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ISGD_fns import *
from torch.nn.utils import clip_grad_norm

# Set seed
torch.manual_seed(1)

# Hyperparameters
BATCH_SIZE = 1
TEST_BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.001
MOMENTUM = 0.0
REGULARIZATION = 0  # 1e-4
SGD_TYPE = 'implicit'  # 'implicit'
OPTIMIZER_TYPE = 'SGD'  # 'Adam', 'RMSprop', 'SGD'
CLIPPING_THRESHOLD = 0  # CLIPPING_THRESHOLD = 0 means no clipping
BATCH_NORM = False
assert (BATCH_SIZE > 1 if BATCH_NORM else True), 'For nn.BatchNorm1d to work, the batch size has to be greater than 1'

# Update learning rate for ISGD
Hyperparameters.set_lr(LEARNING_RATE)
Hyperparameters.set_regularization(REGULARIZATION)
Hyperparameters.set_sgd_type(SGD_TYPE)

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

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = IsgdRelu(320, 50)  # nn.Linear(320, 50)  #
        self.batch_norm = nn.BatchNorm1d(50, affine=False) if BATCH_NORM else IsgdIdentity()
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


# Define optimizer
model = Net()
optimizer = None
assert OPTIMIZER_TYPE in {'SGD', 'RMSprop', 'Adam'}
if OPTIMIZER_TYPE == 'SGD':
    # If implicit then regularization is already done in the backprop,
    # so it shouldn't be included in the optimizer
    if SGD_TYPE == 'implicit':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=REGULARIZATION)

elif OPTIMIZER_TYPE == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters())
elif OPTIMIZER_TYPE == 'Adam':
    optimizer = optim.Adam(model.parameters())


# Define training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # Clip gradients
        # As implemented in https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L162-L164
        if CLIPPING_THRESHOLD != 0:
            clip_grad_norm(model.parameters(), CLIPPING_THRESHOLD)

        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), loss.data[0]))


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
