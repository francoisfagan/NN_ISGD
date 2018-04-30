self.batch_norm = nn.BatchNorm1d(50, affine=False) if Hp.batch_norm else isgd_fns.IsgdIdentity()
# assert (
    #     Hp.batch_size > 1 if Hp.batch_norm else True), 'For nn.BatchNorm1d to work, the batch size has to be greater than 1'
    # batch_norm = None  # True/False indicator of whether to use batch normalization or not

# Hyperpameters for RNN
Hp.train_length = 10000
Hp.test_length = 300
Hp.sequence_length = 11
Hp.input_size = 2
Hp.hidden_size = 10
Hp.output_size = 1

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



# u[i, j] = real_root_closest_to_zero(x[:, i, j])
# cimport cython
# from libc.math cimport sqrt, acos, cos,sin, pow

import math
from functools import reduce

# cdef real_root_closest_to_zero(float [:] coeff):
#     """
#     Given a list of polynomial coefficients,
#     return the real root that is closest to zero
#
#     Args:
#         coeff:  List of polynomial coefficients
#
#     Returns:
#         root_closest_to_zero:   Root that is closest to zero
#
#     """
#     # Calculate all (complex) roots
#     # Could use np.roots(coeff)
#     # However cube_solver.solve(coeff) is faster and more accurate
#
#     roots = solve(coeff)
#     # Note, this doesn't work if there are no real roots, e.g. quadratic
#     # print(np.array(coeff))
#     # print(roots)
#
#     # Extract real roots
#     # Note cannot use root.imag == 0 since numpy sometimes has a tiny imaginary component for real roots
#     # See: https://stackoverflow.com/questions/28081247/print-real-roots-only-in-numpy
#     real_roots = (root.real for root in roots if abs(root.imag) < 1e-10)
#
#     # Extract the real root that is closest to zero
#     root = reduce((lambda x, y: x if (abs(x) < abs(y)) else y), real_roots)
#
#     # Change from double to float
#     # Otherwise the tensor operations are not consistent
#     root = root.astype('float32')
#
#     return root
#
# # Main Function takes in the coefficient of the Cubic Polynomial
# # as parameters and it returns the roots in form of numpy array.
# # Polynomial Structure -> ax^3 + bx^2 + cx + d = 0
#
# # @cython.cdivision(True)
# cdef solve(float [:] coeff):
#     cdef float a = coeff[0]
#     cdef float b = coeff[1]
#     cdef float c = coeff[2]
#     cdef float d = coeff[3]
#
#     if (a == 0 and b == 0):  # Case for handling Liner Equation
#         return np.array([(-d * 1.0) / c])  # Returning linear root as numpy array.
#
#     elif (a == 0):  # Case for handling Quadratic Equations
#
#         D = c * c - 4.0 * b * d  # Helper Temporary Variable
#         if D >= 0:
#             D = sqrt(D)
#             x1 = (-c + D) / (2.0 * b)
#             x2 = (-c - D) / (2.0 * b)
#         else:
#             D = sqrt(-D)
#             x1 = (-c + D * 1j) / (2.0 * b)
#             x2 = (-c - D * 1j) / (2.0 * b)
#
#         return np.array([x1, x2])  # Returning Quadratic Roots as numpy array.
#
#     f = findF(a, b, c)  # Helper Temporary Variable
#     g = findG(a, b, c, d)  # Helper Temporary Variable
#     h = findH(g, f)  # Helper Temporary Variable
#
#     if f == 0 and g == 0 and h == 0:  # All 3 Roots are Real and Equal
#         if (d / a) >= 0:
#             x = (d / (1.0 * a)) ** (1 / 3.0) * -1
#         else:
#             x = (-d / (1.0 * a)) ** (1 / 3.0)
#         return np.array([x, x, x])  # Returning Equal Roots as numpy array.
#
#     elif h <= 0:  # All 3 roots are Real
#
#         i = sqrt(((g ** 2.0) / 4.0) - h)  # Helper Temporary Variable
#         j = i ** (1 / 3.0)  # Helper Temporary Variable
#         k = acos(-(g / (2 * i)))  # Helper Temporary Variable
#         L = j * -1  # Helper Temporary Variable
#         M = cos(k / 3.0)  # Helper Temporary Variable
#         N = sqrt(3) * sin(k / 3.0)  # Helper Temporary Variable
#         P = (b / (3.0 * a)) * -1  # Helper Temporary Variable
#
#         x1 = 2 * j * cos(k / 3.0) - (b / (3.0 * a))
#         x2 = L * (M + N) + P
#         x3 = L * (M - N) + P
#
#         return np.array([x1, x2, x3])  # Returning Real Roots as numpy array.
#
#     elif h > 0:  # One Real Root and two Complex Roots
#         R = -(g / 2.0) + sqrt(h)  # Helper Temporary Variable
#         if R >= 0:
#             S = R ** (1 / 3.0)  # Helper Temporary Variable
#         else:
#             S = (-R) ** (1 / 3.0) * -1  # Helper Temporary Variable
#         T = -(g / 2.0) - sqrt(h)
#         if T >= 0:
#             U = (T ** (1 / 3.0))  # Helper Temporary Variable
#         else:
#             U = ((-T) ** (1 / 3.0)) * -1  # Helper Temporary Variable
#
#         x1 = (S + U) - (b / (3.0 * a))
#         x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * sqrt(3) * 0.5j
#         x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * sqrt(3) * 0.5j
#
#         return np.array([x1, x2, x3])  # Returning One Real Root and two Complex Roots as numpy array.
#
#
# # Helper function to return float value of f.
# # @cython.cdivision(True)
# cdef findF(float a, float b, float c):
#     return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0
#
#
# # Helper function to return float value of g.
# # @cython.cdivision(True)
# cdef findG(float a, float b, float c, float d):
#     return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0
#
#
# # Helper function to return float value of h.
# # @cython.cdivision(True)
# cdef findH(float g, float f):
#     return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

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

@classmethod
def set_hyperparameters(cls, batch_norm, batch_size, clipping_threshold, initialization_scale, lr, mu, sgd_type,
                        test_batch_size):
    cls.batch_norm = batch_norm
    cls.batch_size = batch_size
    cls.clipping_threshold = clipping_threshold
    cls.initialization_scale = initialization_scale
    cls.lr = lr
    cls.mu = mu
    cls.sgd_type = sgd_type
    cls.test_batch_size = test_batch_size





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