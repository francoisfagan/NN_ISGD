def calc_grad_input(input, weight, bias, output, grad_output, sigma):
    """Returns the gradient of the input for the activation function sigma for ISGD
    """
    # return None
    if sigma == 'linear':
        return grad_output.mm(weight)
    elif sigma == 'relu':
        sgn_output = (output >= 0).type(torch.FloatTensor)
        return (grad_output.mul(sgn_output)).mm(weight)




# def a_linear(s, d, c):
#     """
#     Arguments:
#     s [1 x m]      Sign of back-propagated gradient
#     d [1 x m]      Weighted constant, proportional to the sqrt(abs(back-propagated gradient))
#     c [1 x m]      Logit contracted by ridge-regularization
#
#     Return
#     a [1 x m]  Solution of ISGD update for each output
#     """
#     a = - s * d  # Note that this is element-wise multiplication
#     return a


# def a_relu(s, d, c, lr):
#     """
#     Arguments:
#     s [1 x m]      Sign of back-propagated gradient
#     d [1 x m]      Weighted constant, proportional to the sqrt(abs(back-propagated gradient))
#     c [1 x m]      Logit contracted by ridge-regularization
#     lr [1]         Learning rate
#
#     Return
#     alpha [1 x m]  Solution of ISGD update for each output
#     """
#     cond1 = ((s == +1) * (c <= 0)).type(torch.FloatTensor)
#     cond2 = ((s == +1) * (c > 0) * (c <= (lr * d ** 2))).type(torch.FloatTensor)
#     cond3 = ((s == +1) * (c > (lr * d ** 2))).type(torch.FloatTensor)
#     cond4 = ((s == -1) * (c <= -(lr * d ** 2) / 2.0)).type(torch.FloatTensor)
#     cond5 = ((s == -1) * (c > -(lr * d ** 2) / 2.0)).type(torch.FloatTensor)
#
#     a = (0.0 * cond1
#          - (c / (lr * d)) * cond2
#          - d * cond3
#          + 0.0 * cond4
#          + d * cond5
#          )
#
#     # a might contain Nan values if d = 0 at certain elements due to diving by d in (c / (lr * d)) * cond2
#     # The operation below sets all Nans to zero
#     # This is the appropriate value for ISGD
#     a[a != a] = 0
#
#     return a



@staticmethod
@once_differentiable
def backward(ctx, grad_output):
    input, weight, bias, output = ctx.saved_tensors

    # Find all nodes where the output is greater than or equal to 0
    ge0 = (output > 0).type(torch.FloatTensor)  # [1 x m]

    # Mask the back-propagated gradient to zero out elements where the output is zero.
    grad_output_masked = ge0 * grad_output  # [1 x m]

    # Calculate gradients
    grad_input = grad_output_masked.mm(weight)  # [1 x n]
    grad_weight = grad_output_masked.t().mm(input)  # [m x n]
    grad_bias = grad_output_masked.sum(0).squeeze(0)  # [m]

    return grad_input, grad_weight, grad_bias


def fn(a,b,c):
    """

    Args:
        a:
        b:
        c:

    Returns:

    """



## Old version
# class IsgdLinearFunction(torch.autograd.Function):
#
#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, input, weight, bias=None):
#         output = input.mm(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#
#         ctx.save_for_backward(input, weight, bias, output)
#         return output
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         return IsgdUpdate.isgd_update(ctx.saved_tensors, grad_output, 'linear')


## Old
# class IsgdReluFunction(torch.autograd.Function):
#
#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, input, weight, bias=None):
#         output = input.mm(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#
#         output = torch.clamp(output, min=0.0)
#         ctx.save_for_backward(input, weight, bias, output)
#         return output
#
#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         return IsgdUpdate.isgd_update(ctx.saved_tensors, grad_output, 'relu')



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
    if sigma == 'linear':
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

        s = torch.sign(grad_output)
        z_norm = math.sqrt((torch.norm(input) ** 2 + 1.0))
        d = z_norm * math.sqrt(cls.lr / (1.0 + cls.lr * cls.mu)) * torch.sqrt(torch.abs(grad_output))
        c = output / (1.0 + cls.lr * cls.mu)
        a = alpha(s, d, c, sigma)

        new_weight = weight / (1.0 + cls.lr * cls.mu) + (a.mul(d)).t().mm(input) / z_norm ** 2
        grad_weight = (weight - new_weight) / cls.lr

        new_bias = bias / (1.0 + cls.lr * cls.mu) + a.mul(d).squeeze() / z_norm ** 2
        grad_bias = (bias - new_bias) / cls.lr

        grad_input = None #calc_grad_input(input, weight, bias, output, grad_output, sigma)

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




# input, weight, bias, output = ctx.saved_tensors
# grad_input = grad_weight = grad_bias = None
# lr = 0.001
# mu = 0.0
#
# s = torch.sign(grad_output)
# z_norm = math.sqrt((torch.norm(input) ** 2 + 1.0))
# d = z_norm * math.sqrt(lr / (1.0 + lr * mu)) * torch.sqrt(torch.abs(grad_output))
# c = output / (1.0 + lr * mu)
# a = alpha(s, d, c, 'linear')
#
# new_weight = weight / (1.0 + lr * mu) + (a.mul(d)).t().mm(input) / z_norm ** 2
# grad_weight = (weight - new_weight) / lr
#
# new_bias = bias / (1.0 + lr * mu) + a.mul(d).squeeze() / z_norm ** 2
# grad_bias = (bias - new_bias) / lr
#
# # # Original
# # grad_input = grad_output.mm(weight)
# # grad_weight = grad_output.t().mm(input)
# #
# # if bias is not None:
# #     grad_bias = grad_output.sum(0).squeeze(0)
#
# return grad_input, grad_weight, grad_bias