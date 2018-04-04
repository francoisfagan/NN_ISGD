import torch
import torch.nn as nn
import math
from torch.autograd.function import once_differentiable


def alpha(s, d, c, sigma):
    """Returns alpha for the activation function sigma for ISGD
    """
    if sigma == 'linear':
        return -s.mul(d)
    elif sigma == 'relu':
        #     cond1 = (s == 1).mul(c <= 0).type(torch.FloatTensor)
        cond2 = (s == 1).mul(c > 0).mul(c <= d ** 2).type(torch.FloatTensor)
        cond3 = (s == 1).mul(c > d ** 2).type(torch.FloatTensor)
        #     cond4 = (s == -1).mul(c <= -d**2/2.0).type(torch.FloatTensor)
        cond5 = (s == -1).mul(c > -d ** 2 / 2.0).type(torch.FloatTensor)

        return (0.0
                #              + 0.0 * cond1
                - (c.div(d)).mul(cond2)
                - d.mul(cond3)
                #             + 0.0 * cond4
                + d.mul(cond5)
                )

    else:
        raise ValueError('sigma must be in {linear, relu}')


def calc_grad_input(input, weight, bias, output, grad_output, sigma):
    """Returns the gradient of the input for the activation function sigma for ISGD
    """
    # return None
        return grad_output.mm(weight)
    elif sigma == 'relu':
        sgn_output = (output >= 0).type(torch.FloatTensor)
        return (grad_output.mul(sgn_output)).mm(weight)


class IsgdUpdate:
    """Wrapper around alpha update with learning rate and regularization constant"""

    lr = 0.01  # Learning rate
    mu = 0.0  # L-2 regularization constant

    @classmethod
    def set_lr(cls, lr):
        cls.lr = lr

    @classmethod
    def set_regularization(cls, mu):
        cls.mu = mu

    @classmethod
    def isgd_update(cls, saved_tensors, grad_output, sigma):
        input, weight, bias, output = saved_tensors

        # Calculate alpha
        s = torch.sign(grad_output)
        z_norm = math.sqrt((torch.norm(input) ** 2 + 1.0))
        d = z_norm * math.sqrt(cls.lr / (1.0 + cls.lr * cls.mu)) * torch.sqrt(torch.abs(grad_output))
        c = output / (1.0 + cls.lr * cls.mu)
        a = alpha(s, d, c, sigma)

        # Calculate theta = (weight, bias) gradients
        new_weight = weight / (1.0 + cls.lr * cls.mu) + (a.mul(d)).t().mm(input) / z_norm ** 2
        grad_weight = (weight - new_weight) / cls.lr

        new_bias = bias / (1.0 + cls.lr * cls.mu) + a.mul(d).squeeze() / z_norm ** 2
        grad_bias = (bias - new_bias) / cls.lr

        # Calculate input gradient
        grad_input = calc_grad_input(input, weight, bias, output, grad_output, sigma)

        # Return gradients
        return grad_input, grad_weight, grad_bias


class IsgdLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return IsgdUpdate.isgd_update(ctx.saved_tensors, grad_output, 'linear')


class IsgdLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(IsgdLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return IsgdLinearFunction.apply(input, self.weight, self.bias)


class IsgdReluFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        output = torch.clamp(output, min=0.0)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return IsgdUpdate.isgd_update(ctx.saved_tensors, grad_output, 'relu')


class IsgdRelu(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(IsgdRelu, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return IsgdReluFunction.apply(input, self.weight, self.bias)
