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
from torch.autograd.function import once_differentiable
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


def cubic_solution(coeff):
    """
    Given a list of polynomial coefficients,
    return the real root that is closest to zero.

    Based on the code of the cubic root solver of
        Shril Kumar [(shril.iitdhn@gmail.com),(github.com/shril)] &
        Devojoyti Halder [(devjyoti.itachi@gmail.com),(github.com/devojoyti)]
    available:
        https://github.com/shril/CubicEquationSolver

    Args:
        coeff:  List of polynomial coefficients

    Returns:
        Real root closest to zero

    """
    a, b, c, d = coeff  # [b x m] for a, b, c, d

    # Helper functions
    f = ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0  # [b x m]
    g = (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0  # [b x m]
    h = ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)  # [b x m]

    # Calculate conditions for number of roots
    c1 = ((f == 0) * (g == 0) * (h == 0))  # All 3 Roots are Real and Equal
    c2 = (1 - c1) * (h <= 0)  # All 3 roots are Real
    # c3 = (1 - c1) * (h > 0)  # One Real Root and two Complex Roots

    # The final condition c3 is by far the most common and so we make it the default value of u
    # One Real Root and two Complex Roots
    R = -(g / 2.0) + torch.sqrt(torch.abs(h))  # Helper Temporary Variable
    S = torch.abs(R) ** (1 / 3.0) * torch.sign(R)
    T = -(g / 2.0) - torch.sqrt(torch.abs(h))
    U = (torch.abs(T) ** (1 / 3.0)) * torch.sign(T)  # Helper Temporary Variable
    root = (S + U) - (b / (3.0 * a))

    # All 3 Roots are Real and Equal
    if torch.sum(c1) > 0:
        y = d[c1] / (1.0 * a[c1])
        root[c1] = - torch.abs(y) ** (1 / 3.0) * torch.sign(y)

    # All 3 roots are Real
    # The code below assumes that the minimum real root is unique (which seems to be the case)
    if torch.sum(c2) > 0:
        i = torch.sqrt(((g[c2] ** 2.0) / 4.0) + torch.abs(h[c2]))  # Helper Temporary Variable
        j = i ** (1 / 3.0)  # Helper Temporary Variable
        k = torch.acos(-(g[c2] / (2 * i)))  # Helper Temporary Variable
        L = j * -1  # Helper Temporary Variable
        M = torch.cos(k / 3.0)  # Helper Temporary Variable
        N = math.sqrt(3) * torch.sin(k / 3.0)  # Helper Temporary Variable
        P = (b[c2] / (3.0 * a[c2])) * -1  # Helper Temporary Variable

        root_1 = 2 * j * torch.cos(k / 3.0) + P
        root_2 = L * (M + N) + P
        root_3 = L * (M - N) + P

        roots = torch.stack((root_1, root_2, root_3))
        root_abs, _ = torch.min(torch.abs(roots), 0)
        root[c2] = torch.sum((roots == root_abs).float() * root_abs
                             + (roots == -root_abs).float() * -root_abs,
                             0)

    return root


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
        input, weight, bias, output = ctx.saved_tensors
        logit = ctx.logit  # [b x m]

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':

            # Calculate u
            u = grad_output / (1.0 + lr * mu)  # [b x m]

            # Calculate input gradient
            grad_input = grad_output.mm(weight)  # [b x n]

            # Calculate grad_weight, grad_bias
            grad_weight = weight * mu / (1.0 + lr * mu) + u.t().mm(input)  # [m x n]
            grad_bias = bias * mu / (1.0 + lr * mu) + u.t().sum(1)  # [m]

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
        input, weight, bias, output = ctx.saved_tensors
        logit = ctx.logit  # [b x m]

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':

            # Calculate input gradient
            # We do this first as we will want to initialize u = grad_output_masked
            ge0 = (output > 0).float()  # [b x m]
            grad_output_masked = ge0 * grad_output  # [b x m]
            grad_input = grad_output_masked.mm(weight)  # [b x n]

            # ISGD constants
            s = torch.sign(grad_output)  # [b x m]
            z_norm_squared = torch.norm(input, p=2, dim=1) ** 2 + 1.0  # [b]
            c = logit / (1.0 + lr * mu)  # [b x m]

            # Initialize u to be the ESGD relu solution
            # (see ESGD section for more details on the code)
            # This way only a few values might need to be changed
            u = grad_output_masked / (1.0 + lr * mu)  # [b x m]

            # Calculate conditions for u not covered by ESGD
            threshold = lr * torch.mul(z_norm_squared, grad_output.t()).t() / (1.0 + lr * mu)  # [b x m]
            c2 = (s == +1) * (c > 0) * (c <= threshold)  # [b x m]
            c5 = (s == -1) * (c > threshold / 2.0) * (c <= 0)  # [b x m]

            # # Update u according to those conditions
            # if torch.sum(c2) > 0:
            #     # u[c2] = c[c2] / z_norm_squared_mat[c2] / lr
            #     u = torch.where(c2, torch.div(c.t(), z_norm_squared).t() / lr, u)
            # if torch.sum(c5) > 0:
            #     # u[c5] = grad_output[c5] / (1.0 + lr * mu)
            #     u = torch.where(c5, grad_output / (1.0 + lr * mu), u)

            # Calculate grad_weight, grad_bias
            grad_weight = weight * mu / (1.0 + lr * mu) + u.t().mm(input)  # [m x n]
            grad_bias = bias * mu / (1.0 + lr * mu) + u.t().sum(1)  # [m]

        else:  # Explicit SGD update
            # Find all nodes where the output is greater than or equal to 0
            # ge0 = (output > 0).float()  # [b x m]
            ge0 = (output > 0).float()  # [b x m]

            # Mask the back-propagated gradient to zero out elements where the output is zero.
            grad_output_masked = ge0 * grad_output  # [b x m]

            # Calculate gradients
            grad_input = grad_output_masked.mm(weight)  # [b x n]
            grad_weight = grad_output_masked.t().mm(input)  # [m x n]
            grad_bias = grad_output_masked.sum(0).squeeze(0)  # [m]

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
        input, weight, bias, output = ctx.saved_tensors
        logit = ctx.logit  # [b x m]

        # If implicit do the implicit SGD update, else do the explicit SGD update
        if sgd_type == 'implicit':

            # ISGD constants
            b = grad_output / (1 + lr * mu)  # [b x m]
            c = logit / (1.0 + lr * mu)  # [b x m]
            z_norm_squared_mat = (torch.norm(input, p=2, dim=1) ** 2 + 1.0).unsqueeze(1).expand_as(c)  # [b x m]

            # Coefficients of cubic equation for each power:
            # a3*u**3 + a2*u**2 + a1*u + a0 = 0
            a3 = ((lr * z_norm_squared_mat) ** 2)  # [b x m]
            a2 = (-2 * lr * c * z_norm_squared_mat)  # [b x m]
            a1 = (1 + c ** 2)  # [b x m]
            a0 = (- b)  # [b x m]

            # # Coefficients as one big numpy matrix
            # coeff = torch.stack((a3, a2, a1, a0)).cpu().numpy()  # [4 x b x m]
            #
            # # Calculate roots of cubic that are real and closest to zero
            # # Note that this is currently very slow!
            # # This is because np.apply_along_axis implements a python "for loop" and is not optimized
            # # There doesn't seem to be a simple way of improving this
            # # Perhaps in the future Cython could be used to speed up this computation
            # # roots = np.apply_along_axis(real_root_closest_to_zero, 0, coeff)  # [b x m] # Real root closest to zero
            # roots = cubic_root_closest_to_0.get_roots(coeff)
            # u = torch.from_numpy(roots).to(Hp.device)  # [b x m]

            u = cubic_solution([a3, a2, a1, a0])

            # Calculate input gradient
            grad_output_scaled = grad_output / (1 + logit ** 2)  # [b x m]
            grad_input = grad_output_scaled.mm(weight)  # [b x n]

            # Calculate grad_weight, grad_bias
            grad_weight = weight * mu / (1.0 + lr * mu) + u.t().mm(input)  # [m x n]
            grad_bias = bias * mu / (1.0 + lr * mu) + u.t().sum(1)  # [m]


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

        ge0 = (logit > 0).float()  # [b x m]
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
        ge0 = (output > 0).float()  # [b x m]
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
