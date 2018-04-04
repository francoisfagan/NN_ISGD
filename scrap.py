
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