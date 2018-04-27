""" This file defines all of the activation, layers and functions
for implicit stochastic gradient descent on neural networks

It is helpful to keep track of dimensions while doing matrix and vector operations.
We use the following shorthand to help keep track:

    b = mini-batch size
    n = input size to neural network layer
    m = output size from neural network layer

If you want to implement a new activation function and layer all you have to do is:
    1. Copy, rename and edit IsgdTemplateFunction so that it can do
        forward and backward passes with your activation function
    2. Copy and edit IsgdTemplate
Details of how to edit are given in the IsgdTemplateFunction and IsgdTemplate classes

"""

import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd.function import once_differentiable
from functools import reduce
from utils import Hp


# Utility functions
def calc_logit(input, weight, bias=None):
    """
    Calculate logit = input.mm(weight.t()) + bias

    Args:
        input:  [b x n]         Input vector
        weight:  [m x n]        Weight matrix
        bias:  [m]              Bias vector

    Returns:
        logit: [b x n]          Logit = input.mm(weight.t()) + bias

    """

    logit = input.mm(weight.t())
    if bias is not None:
        logit += bias.unsqueeze(0).expand_as(logit)

    return logit


def calc_backwards_variables(saved_tensors, logit, grad_output, lr, mu):
    """
    Calculate the variables required for back-propagation
    
    Args:
        saved_tensors:          Stores from forward-propagation the input, weight, bias, output
        logit: [b x n]          Stores from forward-propagation the logit
        grad_output: [b x m]    The gradient that has been back-propagated to this layer
        lr: [1]                 Learning rate
        mu: [1]                 Ridge-regularization constant

    Returns:
        input: [b x n]          Input vector
        weight: [m x n]         Weight matrix
        bias: [m]               Bias vector
        output [b x m]          Input to the next layer = logit put through the non-linear activation function
        logit: [b x n]          Logit
        s: [b x m]              Sign of back-propagated gradient
        z_norm: [b]             2-norm of (input, 1)
        d: [b x m]              Weighted constant, proportional to the sqrt(abs(back-propagated gradient))
        c: [b x m]              Logit contracted by ridge-regularization
    """

    # Unpack saved values
    input, weight, bias, output = saved_tensors

    # ISGD constants
    s = torch.sign(grad_output)  # [b x m]
    z_norm = torch.sqrt(torch.norm(input, p=2, dim=1) ** 2 + 1.0)  # [b]
    d = torch.mul(z_norm, torch.sqrt(torch.abs(grad_output)).t()).t() / math.sqrt(1.0 + lr * mu)  # [b x m]
    c = logit / (1.0 + lr * mu)  # [b x m]

    return input, weight, bias, output, logit, s, z_norm, d, c


def calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias):
    """
    Calculate the gradient of the weight matrix and bias vector

    Args:
        weight: [m x n]         Weight matrix
        mu: [1]                 Ridge-regularization constant
        lr: [1]                 Learning rate
        a: [b x m]              Solution of ISGD update
        d: [b x m]              Weighted constant, proportional to the sqrt(abs(back-propagated gradient))
        input: [b x n]          Input vector
        z_norm: [b]             2-norm of (input, 1)
        bias: [m]               Bias vector

    Returns:
        grad_weight: [m x n]    Gradient of the weight matrix
        grad_bias: [m]          Gradient of the bias vector

    """

    grad_weight = weight * mu / (1.0 + lr * mu) - torch.mul(z_norm ** -2, (a * d).t()).mm(input)  # [m x n]
    grad_bias = bias * mu / (1.0 + lr * mu) - torch.mul(z_norm ** -2, (a * d).t()).sum(1)  # [m]
    return grad_weight, grad_bias


def real_root_closest_to_zero(coeff):
    """
    Given a list of polynomial coefficients,
    return the real root that is closest to zero

    Args:
        coeff:  List of polynomial coefficients

    Returns:
        root_closest_to_zero:   Root that is closest to zero

    """
    roots = np.roots(coeff)
    real_roots = [root.real for root in roots if root.imag == 0]
    root_closest_to_zero = reduce((lambda x, y: x if (abs(x) < abs(y)) else y), real_roots)
    return root_closest_to_zero


# Function classes for activation functions
class IsgdTemplateFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [b x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [b x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [b x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)  # [b x m]

        # Non-linear activation function
        output = None  # THIS NEEDS TO BE FILLED IN

        # Save context for back-propagation
        ctx.save_for_backward(input, weight, bias, output)
        ctx.logit = logit

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Calculate back-propagation gradients for input, weight and bias

        Args:
            ctx:                    Context save from the forward-propagation
            grad_output: [b x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [b x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu, sgd_type = Hp.get_isgd_hyperparameters()
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':

            # Calculate a
            a = None  # THIS NEEDS TO BE FILLED IN

            # Calculate input gradient
            grad_input = None  # THIS NEEDS TO BE FILLED IN

            # Calculate grad_weight, grad_bias and return all gradients
            grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

        else:  # Explicit SGD update
            grad_input, grad_weight, grad_bias = None, None, None

        return grad_input, grad_weight, grad_bias


class IsgdLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [b x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [b x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [b x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)  # [b x m]

        # Non-linear activation function
        output = logit  # [b x m]

        # Save context for back-propagation
        ctx.save_for_backward(input, weight, bias, output)
        ctx.logit = logit

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Calculate back-propagation gradients for input, weight and bias

        Args:
            ctx:                    Context save from the forward-propagation
            grad_output: [b x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [b x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu, sgd_type = Hp.get_isgd_hyperparameters()
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':
            # Calculate a
            a = - s * d  # [b x m]

            # Calculate input gradient
            grad_input = grad_output.mm(weight)  # [b x n]

            # Calculate grad_weight, grad_bias and return all gradients
            grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

        else:  # Explicit SGD update
            grad_input = grad_output.mm(weight)  # [b x n]
            grad_weight = grad_output.t().mm(input)  # [m x n]
            grad_bias = grad_output.sum(0).squeeze(0)  # [m]

        return grad_input, grad_weight, grad_bias


class IsgdReluFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [b x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [b x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [b x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)  # [b x m]

        # Non-linear activation function
        output = torch.clamp(logit, min=0.0)  # [b x m]

        # Save context for back-propagation
        ctx.save_for_backward(input, weight, bias, output)
        ctx.logit = logit

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Calculate back-propagation gradients for input, weight and bias

        Args:
            ctx:                    Context save from the forward-propagation
            grad_output: [b x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [b x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu, sgd_type = Hp.get_isgd_hyperparameters()
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':

            # Calculate a
            # Calculate conditions for a
            conds0 = (s == 0).type(torch.FloatTensor)  # [b x m]
            cond1 = ((s == +1) * (c <= 0)).type(torch.FloatTensor)  # [b x m]
            cond2 = ((s == +1) * (c > 0) * (c <= (lr * d ** 2))).type(torch.FloatTensor)  # [b x m]
            cond3 = ((s == +1) * (c > (lr * d ** 2))).type(torch.FloatTensor)  # [b x m]
            cond4 = ((s == -1) * (c <= -(lr * d ** 2) / 2.0)).type(torch.FloatTensor)  # [b x m]
            cond5 = ((s == -1) * (c > -(lr * d ** 2) / 2.0)).type(torch.FloatTensor)  # [b x m]

            # Check that exactly one condition satisfied for each node
            cond_sum = (conds0 + cond1 + cond2 + cond3 + cond4 + cond5)  # [b x m]
            assert torch.mean(
                (cond_sum == 1).type(torch.FloatTensor)) == 1.0, 'No implicit update condition was satisfied'

            # Calculate a
            a = (0.0 * conds0
                 + 0.0 * cond1
                 - (c / (lr * d)) * cond2
                 - d * cond3
                 + 0.0 * cond4
                 + d * cond5
                 )  # [b x m]

            # a might contain Nan values if d = 0 at certain elements due to diving by d in (c / (lr * d)) * cond2
            # The operation below sets all Nans to zero
            # This is the appropriate value for ISGD
            a[a != a] = 0

            # Calculate input gradient
            ge0 = (output > 0).type(torch.FloatTensor)  # [b x m]
            grad_output_masked = ge0 * grad_output  # [b x m]
            grad_input = grad_output_masked.mm(weight)  # [b x n]

            # Calculate grad_weight, grad_bias and return all gradients
            grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

        else:  # Explicit SGD update
            # Find all nodes where the output is greater than or equal to 0
            ge0 = (output > 0).type(torch.FloatTensor)  # [b x m]

            # Mask the back-propagated gradient to zero out elements where the output is zero.
            grad_output_masked = ge0 * grad_output  # [b x m]

            # Calculate gradients
            grad_input = grad_output_masked.mm(weight)  # [b x n]
            grad_weight = grad_output_masked.t().mm(input)  # [m x n]
            grad_bias = grad_output_masked.sum(0).squeeze(0)  # [m]

        return grad_input, grad_weight, grad_bias


class IsgdHardTanhFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [b x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [b x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [b x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)  # [b x m]

        # Non-linear activation function
        output = torch.clamp(logit, min=-1.0, max=1)  # [b x m]

        # Save context for back-propagation
        ctx.save_for_backward(input, weight, bias, output)
        ctx.logit = logit

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Calculate back-propagation gradients for input, weight and bias

        Args:
            ctx:                    Context save from the forward-propagation
            grad_output: [b x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [b x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu, sgd_type = Hp.get_isgd_hyperparameters()
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)
        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':

            # Calculate a
            sc = s * c  # [b x m]
            cond1 = ((sc <= -1)
                     + (sc >= (1 + lr * d ** 2 / 2))).type(torch.FloatTensor)  # [b x m]
            cond2 = ((sc >= (lr * d ** 2 - 1)) * (sc <= 1)
                     + ((sc >= torch.clamp(lr * d ** 2 / 2 - 1, min=1))
                        * (sc <= (1 + lr * d ** 2 / 2)))).type(torch.FloatTensor)  # [b x m]
            cond3 = ((sc >= -1) * (sc <= torch.clamp(lr * d ** 2 - 1, max=1))
                     + (sc >= 1) * (sc <= (lr * d ** 2 / 2 - 1))).type(torch.FloatTensor)  # [b x m]

            # Check that at least one condition satisfied for each node
            # (it is possible for multiple conditions to be satisfied)
            cond_sum = (cond1 + cond2 + cond3)  # [b x m]
            assert (torch.mean((cond_sum >= 1).type(torch.FloatTensor)) == 1.0)

            # If multiple conditions satisfied then normalize
            cond1 = cond1 / cond_sum  # [b x m]
            cond2 = cond2 / cond_sum  # [b x m]
            cond2 = cond2 / cond_sum  # [b x m]

            # Calculate a
            a = (0.0 * cond1
                 - s * d * cond2
                 - (1 + sc) / (lr * d) * cond3
                 )  # [b x m]

            # a might contain Nan values if d = 0 at certain elements due to diving by d in (c / (lr * d)) * cond2
            # The operation below sets all Nans to zero
            # This is the appropriate value for ISGD
            a[a != a] = 0

            # Calculate grad_weight, grad_bias and return all gradients
            grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

            # Calculate input gradient
            non_clamped = ((output > -1) * (output < 1)).type(torch.FloatTensor)  # [b x m]
            grad_output_masked = non_clamped * grad_output  # [b x m]
            grad_input = grad_output_masked.mm(weight)  # [b x n]

        else:  # Explicit SGD update
            non_clamped = ((output > -1) * (output < 1)).type(torch.FloatTensor)  # [b x m]
            grad_output_masked = non_clamped * grad_output  # [b x m]

            # Calculate gradients
            grad_input = grad_output_masked.mm(weight)  # [b x n]
            grad_weight = grad_output_masked.t().mm(input)  # [m x n]
            grad_bias = grad_output_masked.sum(0).squeeze(0)  # [m]

        # Calculate gradients
        return grad_input, grad_weight, grad_bias


class IsgdArctanFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [b x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [b x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [b x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)  # [b x m]

        # Non-linear activation function
        output = torch.atan(logit)  # [b x m]

        # Save context for back-propagation
        ctx.save_for_backward(input, weight, bias, output)
        ctx.logit = logit

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Calculate back-propagation gradients for input, weight and bias

        Args:
            ctx:                    Context save from the forward-propagation
            grad_output: [b x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [b x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu, sgd_type = Hp.get_isgd_hyperparameters()
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':
            # Calculate a
            # # First method using algebraic root finding
            # # It is slow since it needs to move from pytorch to numpy and back
            # # This method as currently coded only works with batch size = 1
            # coeff = np.array([((lr * d) ** 2).numpy()[0],
            #                   (2 * lr * d * c).numpy()[0],
            #                   (c ** 2 + 1).numpy()[0],
            #                   (s * d).numpy()[0]])
            #
            # root_closest_to_zero = np.apply_along_axis(real_root_closest_to_zero, 0, coeff)
            # a = torch.from_numpy(root_closest_to_zero).unsqueeze(1).t().type(torch.FloatTensor)

            # # Second method is iterative
            # # It stays in pytorch, so is faster
            # a = d * 0  # [b x m]
            # a_diff = 1  # Norm difference between previous and current a values
            # iter_count = 0  # Count of number of a iterations
            # while a_diff > 1e-10:
            #     a_new = - s * d / (1.0 + (lr * d * a + c) ** 2)  # [b x m]
            #     a_diff = torch.norm(a - a_new)
            #     a = a_new  # [b x m]
            #     iter_count += 1
            #     if iter_count >= 50:
            #         assert (iter_count < 50), 'Arctan update has failed to converge'

            # Second method is iterative
            # It stays in pytorch, so is faster
            # Make everything doubles to prevent rounding errors
            d_d = d.double()
            s_m_d = (s * d).double()
            c_d = c.double()

            a = d_d * 0  # [b x m]
            a_diff = 1  # Norm difference between previous and current a values
            iter_count = 0  # Count of number of a iterations
            while a_diff > 1e-12:
                a_new = - s_m_d / (1.0 + (lr * d_d * a + c_d) ** 2)  # [b x m]
                a_diff = torch.norm(a - a_new)
                a = a_new  # [b x m]
                iter_count += 1
                if iter_count >= 50:
                    assert (iter_count < 50), 'Arctan update has failed to converge'

            # Make a float so that can be operated on with other tensors
            a = a.float()

            # Calculate grad_weight, grad_bias and return all gradients
            grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

            # Calculate input gradient
            grad_output_scaled = grad_output / (1 + logit ** 2)  # [b x m]
            grad_input = grad_output_scaled.mm(weight)  # [b x n]

        else:  # Explicit SGD update
            grad_output_scaled = grad_output / (1 + logit ** 2)  # [b x m]
            grad_input = grad_output_scaled.mm(weight)  # [b x n]
            grad_weight = grad_output_scaled.t().mm(input)  # [m x n]
            grad_bias = grad_output_scaled.sum(0).squeeze(0)  # [m]

        return grad_input, grad_weight, grad_bias


class IsgdSeluFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [b x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [b x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [b x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)  # [b x m]

        # Non-linear activation function
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        ge0 = (logit > 0).type(torch.FloatTensor)  # [b x m]
        output = scale * (ge0 * logit
                          + (1 - ge0) * alpha * (torch.exp(logit) - 1)
                          )  # [b x m]

        # Save context for back-propagation
        ctx.save_for_backward(input, weight, bias, output)
        ctx.logit = logit

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Calculate back-propagation gradients for input, weight and bias

        Args:
            ctx:                    Context save from the forward-propagation
            grad_output: [b x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [b x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu, sgd_type = Hp.get_isgd_hyperparameters()
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # If implicit do the implicit SGD update, else do the explicit SGD update
        assert sgd_type != 'implicit', 'Implicit SGD not yet available for selu'

        # Calculate selu multiplied grad_output
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        ge0 = (output > 0).type(torch.FloatTensor)  # [b x m]
        selu_multipier = scale * (ge0 + (1 - ge0) * alpha * torch.exp(output))  # [b x m]
        selu_multiplied_grad_output = selu_multipier * grad_output  # [b x m]

        # Calculate gradients
        grad_input = selu_multiplied_grad_output.mm(weight)  # [b x n]
        grad_weight = selu_multiplied_grad_output.t().mm(input)  # [m x n]
        grad_bias = selu_multiplied_grad_output.sum(0).squeeze(0)  # [m]

        return grad_input, grad_weight, grad_bias


# Modules for layers with particular activation functions
class IsgdTemplate(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and output features
        as well as creating and initializing the parameters

        Args:
            input_features: [b x n]     Input features from previous layer
            output_features: [b x m]    Output features for subsequent layer
            bias:
        """
        super(IsgdTemplate, self).__init__()

        # Store features
        self.input_features = input_features
        self.output_features = output_features

        # Create parameters
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        # Not a very smart way to initialize weights
        bias_scale, weight_scale = Hp.get_initialization_scale(input_features, output_features)
        self.weight.data.uniform_(- weight_scale, weight_scale)
        if bias is not None:
            self.bias.data.uniform_(-bias_scale, bias_scale)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [b x m]      Input features from previous layer

        Returns:
            output: [b x n]     Output features for subsequent layer
        """
        output = IsgdTemplateFunction.apply(input, self.weight, self.bias)
        return output


class IsgdLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and output features
        as well as creating and initializing the parameters

        Args:
            input_features: [b x n]     Input features from previous layer
            output_features: [b x m]    Output features for subsequent layer
            bias:
        """
        super(IsgdLinear, self).__init__()

        # Store features
        self.input_features = input_features
        self.output_features = output_features

        # Create parameters
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [b x m]      Input features from previous layer

        Returns:
            output: [b x n]     Output features for subsequent layer
        """
        output = IsgdLinearFunction.apply(input, self.weight, self.bias)
        return output


class IsgdRelu(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and output features
        as well as creating and initializing the parameters

        Args:
            input_features: [b x n]     Input features from previous layer
            output_features: [b x m]    Output features for subsequent layer
            bias:
        """
        super(IsgdRelu, self).__init__()

        # Store features
        self.input_features = input_features
        self.output_features = output_features

        # Create parameters
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [b x m]      Input features from previous layer

        Returns:
            output: [b x n]     Output features for subsequent layer
        """
        output = IsgdReluFunction.apply(input, self.weight, self.bias)
        return output


class IsgdHardTanh(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and output features
        as well as creating and initializing the parameters

        Args:
            input_features: [b x n]     Input features from previous layer
            output_features: [b x m]    Output features for subsequent layer
            bias:
        """
        super(IsgdHardTanh, self).__init__()

        # Store features
        self.input_features = input_features
        self.output_features = output_features

        # Create parameters
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [b x m]      Input features from previous layer

        Returns:
            output: [b x n]     Output features for subsequent layer
        """
        output = IsgdHardTanhFunction.apply(input, self.weight, self.bias)
        return output


class IsgdArctan(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and output features
        as well as creating and initializing the parameters

        Args:
            input_features: [b x n]     Input features from previous layer
            output_features: [b x m]    Output features for subsequent layer
            bias:
        """
        super(IsgdArctan, self).__init__()

        # Store features
        self.input_features = input_features
        self.output_features = output_features

        # Create parameters
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [b x m]      Input features from previous layer

        Returns:
            output: [b x n]     Output features for subsequent layer
        """
        output = IsgdArctanFunction.apply(input, self.weight, self.bias)
        return output


class IsgdSelu(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and output features
        as well as creating and initializing the parameters

        Args:
            input_features: [b x n]     Input features from previous layer
            output_features: [b x m]    Output features for subsequent layer
            bias:
        """
        super(IsgdSelu, self).__init__()

        # Store features
        self.input_features = input_features
        self.output_features = output_features

        # Create parameters
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [b x m]      Input features from previous layer

        Returns:
            output: [b x n]     Output features for subsequent layer
        """
        output = IsgdSeluFunction.apply(input, self.weight, self.bias)
        return output


class IsgdIdentity(nn.Module):
    def forward(self, inputs):
        return inputs
