import torch
import torch.nn as nn
import math
from torch.autograd.function import once_differentiable
import numpy as np
from functools import reduce


def calc_logit(input, weight, bias=None):
    """
    Calculate logit = input.mm(weight.t()) + bias

    Args:
        input:  [1 x n]         Input vector
        weight:  [m x n]        Weight matrix
        bias:  [m]              Bias vector

    Returns:
        logit: [1 x n]          Logit = input.mm(weight.t()) + bias

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
        logit: [1 x n]          Stores from forward-propagation the logit
        grad_output: [1 x m]    The gradient that has been back-propagated to this layer
        lr: [1]                 Learning rate
        mu: [1]                 Ridge-regularization constant

    Returns:
        input: [1 x n]          Input vector
        weight: [m x n]         Weight matrix
        bias: [m]               Bias vector
        output [1 x m]          Input to the next layer = logit put through the non-linear activation function
        logit: [1 x n]          Logit
        s: [1 x m]              Sign of back-propagated gradient
        z_norm: [1]             2-norm of (input, 1)
        d: [1 x m]              Weighted constant, proportional to the sqrt(abs(back-propagated gradient))
        c: [1 x m]              Logit contracted by ridge-regularization
    """

    # Unpack saved values
    input, weight, bias, output = saved_tensors

    # ISGD constants
    s = torch.sign(grad_output)
    z_norm = math.sqrt((torch.norm(input) ** 2 + 1.0))
    d = z_norm / math.sqrt(1.0 + lr * mu) * torch.sqrt(torch.abs(grad_output))
    c = logit / (1.0 + lr * mu)

    return input, weight, bias, output, logit, s, z_norm, d, c


def calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias):
    """
    Calculate the gradient of the weight matrix and bias vector

    Args:
        weight: [m x n]         Weight matrix
        mu: [1]                 Ridge-regularization constant
        lr: [1]                 Learning rate
        a: [1 x m]              Solution of ISGD update
        d: [1 x m]              Weighted constant, proportional to the sqrt(abs(back-propagated gradient))
        input: [1 x n]          Input vector
        z_norm: [1]             2-norm of (input, 1)
        bias: [m]               Bias vector

    Returns:
        grad_weight: [m x n]    Gradient of the weight matrix
        grad_bias: [m]          Gradient of the bias vector

    """

    grad_weight = weight * mu / (1.0 + lr * mu) - (a * d).t().mm(input) / z_norm ** 2
    grad_bias = bias * mu / (1.0 + lr * mu) - (a * d).squeeze() / z_norm ** 2
    return grad_weight, grad_bias


def real_root_closest_to_zero(coeff):
    roots = np.roots(coeff)
    real_roots = [root.real for root in roots if root.imag == 0]
    root_closest_to_zero = reduce((lambda x, y: x if (abs(x) < abs(y)) else y), real_roots)
    return root_closest_to_zero


class HyperparameterClass:
    """Stores and sets the learning rate and ridge regularization constant"""

    lr = 0.01  # Learning rate
    mu = 0.0  # Ridge regularization constant

    @classmethod
    def set_lr(cls, lr):
        cls.lr = lr

    @classmethod
    def set_regularization(cls, mu):
        cls.mu = mu


# Functions
class IsgdTemplateFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        """
        Calculate forward-propagation and store variables need for back-propagation

        Args:
            ctx:
            input: [1 x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [1 x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [1 x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)

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
            grad_output: [1 x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [1 x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu = HyperparameterClass.lr, HyperparameterClass.mu
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # Calculate a
        a = None  # THIS NEEDS TO BE FILLED IN

        # Calculate input gradient
        grad_input = None  # THIS NEEDS TO BE FILLED IN

        # Calculate grad_weight, grad_bias and return all gradients
        grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)
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
            input: [1 x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [1 x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [1 x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)

        # Non-linear activation function
        output = logit

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
            grad_output: [1 x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [1 x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu = HyperparameterClass.lr, HyperparameterClass.mu
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # Calculate a
        a = - s * d

        # Calculate input gradient
        grad_input = grad_output.mm(weight)

        # Calculate grad_weight, grad_bias and return all gradients
        grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)
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
            input: [1 x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [1 x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [1 x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)

        # Non-linear activation function
        output = torch.clamp(logit, min=0.0)  # [1 x m]

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
            grad_output: [1 x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [1 x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu = HyperparameterClass.lr, HyperparameterClass.mu
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # Calculate a
        # Calculate conditions for a
        cond1 = ((s == +1) * (c <= 0)).type(torch.FloatTensor)
        cond2 = ((s == +1) * (c > 0) * (c <= (lr * d ** 2))).type(torch.FloatTensor)
        cond3 = ((s == +1) * (c > (lr * d ** 2))).type(torch.FloatTensor)
        cond4 = ((s == -1) * (c <= -(lr * d ** 2) / 2.0)).type(torch.FloatTensor)
        cond5 = ((s == -1) * (c > -(lr * d ** 2) / 2.0)).type(torch.FloatTensor)

        # Check that exactly one condition satisfied for each node
        cond_sum = (cond1 + cond2 + cond3 + cond4 + cond5)
        assert (torch.mean((cond_sum == 1).type(torch.FloatTensor)) == 1.0)

        # Calculate a
        a = (0.0 * cond1
             - (c / (lr * d)) * cond2
             - d * cond3
             + 0.0 * cond4
             + d * cond5
             )

        # a might contain Nan values if d = 0 at certain elements due to diving by d in (c / (lr * d)) * cond2
        # The operation below sets all Nans to zero
        # This is the appropriate value for ISGD
        a[a != a] = 0

        # Calculate input gradient
        ge0 = (output > 0).type(torch.FloatTensor)  # [1 x m]
        grad_output_masked = ge0 * grad_output  # [1 x m]
        grad_input = grad_output_masked.mm(weight)

        # Calculate grad_weight, grad_bias and return all gradients
        grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)
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
            input: [1 x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [1 x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [1 x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)

        # Non-linear activation function
        output = torch.clamp(logit, min=-1.0, max=1)  # [1 x m]

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
            grad_output: [1 x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [1 x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu = HyperparameterClass.lr, HyperparameterClass.mu
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # Calculate a
        sc = s * c
        cond1 = ((sc <= -1)
                 + (sc >= (1 + lr * d ** 2 / 2))).type(torch.FloatTensor)
        cond2 = ((sc >= (lr * d ** 2 - 1)) * (sc <= 1)
                 + ((sc >= torch.clamp(lr * d ** 2 / 2 - 1, min=1))
                    * (sc <= (1 + lr * d ** 2 / 2)))).type(torch.FloatTensor)
        cond3 = ((sc >= -1) * (sc <= torch.clamp(lr * d ** 2 - 1, max=1))
                 + (sc >= 1) * (sc <= (lr * d ** 2 / 2 - 1))).type(torch.FloatTensor)

        # Check that at least one condition satisfied for each node
        # (it is possible for multiple conditions to be satisfied)
        cond_sum = (cond1 + cond2 + cond3)
        assert (torch.mean((cond_sum >= 1).type(torch.FloatTensor)) == 1.0)

        # If multiple conditions satisfied then normalize
        cond1 = cond1 / cond_sum
        cond2 = cond2 / cond_sum
        cond2 = cond2 / cond_sum

        # Calculate a
        a = (0.0 * cond1
             - s * d * cond2
             - (1 + sc) / (lr * d) * cond3
             )

        # a might contain Nan values if d = 0 at certain elements due to diving by d in (c / (lr * d)) * cond2
        # The operation below sets all Nans to zero
        # This is the appropriate value for ISGD
        a[a != a] = 0

        # Calculate grad_weight, grad_bias and return all gradients
        grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

        # Calculate input gradient
        non_clamped = ((output > -1) * (output < 1)).type(torch.FloatTensor)  # [1 x m]
        grad_output_masked = non_clamped * grad_output  # [1 x m]
        grad_input = grad_output_masked.mm(weight)  # [1 x n]

        # Calculate gradients
        # grad_weight = grad_output_masked.t().mm(input)  # [m x n]
        # grad_bias = grad_output_masked.sum(0).squeeze(0)  # [m]
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
            input: [1 x n]      Input vector
            weight: [m x n]     Weight matrix
            bias: [m]           Bias vector

        Returns:
            output [1 x m]     Input to the next layer = logit put through the non-linear activation function
        """

        # Calculate logit [1 x m], where logit = input.mm(weight.t()) + bias
        logit = calc_logit(input, weight, bias)

        # Non-linear activation function
        output = torch.atan(logit)  # THIS NEEDS TO BE FILLED IN

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
            grad_output: [1 x m]    The gradient that has been back-propagated to this layer

        Returns:
            grad_input: [1 x n]     Gradient of the input vector
            grad_weight: [m x n]    Gradient of the weight matrix
            grad_bias: [m]          Gradient of the bias vector
        """

        # Retrieve parameters required for the update
        lr, mu = HyperparameterClass.lr, HyperparameterClass.mu
        input, weight, bias, output, logit, s, z_norm, d, c = calc_backwards_variables(ctx.saved_tensors, ctx.logit,
                                                                                       grad_output, lr, mu)

        # Calculate a
        # # First method using algebraic root finding
        # # It is slow since it needs to move from pytorch to numpy and back
        # coeff = np.array([((lr * d) ** 2).numpy()[0],
        #                   (2 * lr * d * c).numpy()[0],
        #                   (c ** 2 + 1).numpy()[0],
        #                   (s * d).numpy()[0]])
        #
        # root_closest_to_zero = np.apply_along_axis(real_root_closest_to_zero, 0, coeff)
        # a = torch.from_numpy(root_closest_to_zero).unsqueeze(1).t().type(torch.FloatTensor)

        # Second method is iterative
        # It stays in pytorch, so is faster
        a = d * 0
        a_diff = 1  # Norm difference between previous and current a values
        iter_count = 0  # Count of number of a iterations
        while a_diff < 1e-10:
            a_new = - s * d / (1.0 + (lr * d * a + c) ** 2)
            a_diff = torch.norm(a - a_new)
            a = a_new
            iter_count += 1
            assert (iter_count < 20), 'Arctan update has failed to converge'

        # Calculate grad_weight, grad_bias and return all gradients
        grad_weight, grad_bias = calc_weigh_bias_grad(weight, mu, lr, a, d, input, z_norm, bias)

        # Calculate input gradient
        grad_output_scaled = grad_output / (1 + logit ** 2)  # [1 x m]
        grad_input = grad_output_scaled.mm(weight)  # [1 x n]

        return grad_input, grad_weight, grad_bias


# Modules
class IsgdTemplate(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and outputfeatures
        as well as creating and initializing the parameters

        Args:
            input_features: [1 x n]     Input features from previous layer
            output_features: [1 x m]    Output features for subsequent layer
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
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        Set forward and backward-propagation functions.
        Even though this function is called "forward",
        the class which it calls defines both forward and backward-propagations

        Args:
            input: [1 x m]      Input features from previous layer

        Returns:
            output: [1 x n]     Output features for subsequent layer
        """
        output = IsgdTemplateFunction.apply(input, self.weight, self.bias)
        return output


class IsgdLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and outputfeatures
        as well as creating and initializing the parameters

        Args:
            input_features: [1 x n]     Input features from previous layer
            output_features: [1 x m]    Output features for subsequent layer
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
            input: [1 x m]      Input features from previous layer

        Returns:
            output: [1 x n]     Output features for subsequent layer
        """
        output = IsgdLinearFunction.apply(input, self.weight, self.bias)
        return output


class IsgdRelu(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and outputfeatures
        as well as creating and initializing the parameters

        Args:
            input_features: [1 x n]     Input features from previous layer
            output_features: [1 x m]    Output features for subsequent layer
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
            input: [1 x m]      Input features from previous layer

        Returns:
            output: [1 x n]     Output features for subsequent layer
        """
        output = IsgdReluFunction.apply(input, self.weight, self.bias)
        return output


class IsgdHardTanh(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and outputfeatures
        as well as creating and initializing the parameters

        Args:
            input_features: [1 x n]     Input features from previous layer
            output_features: [1 x m]    Output features for subsequent layer
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
            input: [1 x m]      Input features from previous layer

        Returns:
            output: [1 x n]     Output features for subsequent layer
        """
        output = IsgdHardTanhFunction.apply(input, self.weight, self.bias)
        return output


class IsgdArctan(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the module by storing the input and outputfeatures
        as well as creating and initializing the parameters

        Args:
            input_features: [1 x n]     Input features from previous layer
            output_features: [1 x m]    Output features for subsequent layer
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
            input: [1 x m]      Input features from previous layer

        Returns:
            output: [1 x n]     Output features for subsequent layer
        """
        output = IsgdArctanFunction.apply(input, self.weight, self.bias)
        return output
