""""
Demonstrate custom implementation of forward pass and backward propagation
of a BatchNorm-Sigmoid-MSE neural network
"""
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import numpy as np

# custom implementation using Torch tensors
flag_custom_implement_torch = 1
flag_custom_implement_numpy = 1
flag_torch_built_ins_as_ref = 1

# prepare the inputs and targets data
eps = 1e-5
torch.manual_seed(1234)
nSamples = 5
inCh = 3
inImgDepth, inImgRows, inImgCols = 2, 3, 3
input_x = torch.randint(0, 100, (nSamples, inCh, inImgDepth, inImgRows, inImgCols), dtype=torch.float64)
input_weight = torch.rand(inCh, requires_grad=True, dtype=torch.float64)
input_bias = torch.rand(inCh, requires_grad=True, dtype=torch.float64) * 2.0 - 1.0
targets = torch.rand(input_x.shape, dtype=torch.float64) * 2.0 - 1.0

'''
Custom implementation #A: implementation of the forward and backward passes of
BatchNorm-sigmoid-MSE using torch tensors and Function
'''
class MyBatchNormNd(Function):
    """
    Implement our own custom autograd Functions of BatchNorm by subclassing
    torch.autograd.Function and override the forward and backward passes
    """
    @staticmethod
    def forward(ctx, in_x, gamma=None, beta=None):
        """
        override the forward function
        """
        ax = [0] + list(range(2, in_x.dim()))
        batch_mean = in_x.mean(dim=ax, keepdim=True)
        batch_variance_sqrt = torch.sqrt(in_x.var(axis=ax, unbiased=False, keepdim=True) + eps)
        out_y = (in_x - batch_mean) / batch_variance_sqrt

        if gamma is not None:
            out_y *= gamma.reshape(batch_mean.shape)
        if beta is not None:
            out_y += beta.reshape(batch_mean.shape)

        # cache these objects for use in the backward pass
        ctx.save_for_backward(in_x, gamma, beta, batch_mean, batch_variance_sqrt, out_y)

        return out_y

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """
        override the backward function. It receives a Tensor containing the gradient of the loss
        with respect to the output of the custom forward pass, and calculates the gradients of the loss
        with respect to each of the inputs of the custom forward pass.
        """
        print('Performing custom backward of BatchNorm')
        in_x, gamma, beta, batch_mean, batch_variance_sqrt, out_y = ctx.saved_tensors
        ax = [0] + list(range(2, in_x.dim()))
        nelem = in_x.numel() // in_x.size()[1]

        w = torch.ones(batch_mean.shape)
        if gamma is not None:
            w = gamma.reshape(batch_mean.shape)
        b = torch.zeros(batch_mean.shape)
        if beta is not None:
            b = beta.reshape(batch_mean.shape)

        # calculate the gradients of Loss w.r.t. in_x, and the parameters of gamma and beta if not None
        grad_x = w * grad_from_upstream / batch_variance_sqrt \
                 - w * grad_from_upstream.sum(dim=ax, keepdim=True) / (nelem * batch_variance_sqrt) \
                 - (out_y-b) * (grad_from_upstream * (out_y-b)).sum(dim=ax, keepdim=True) \
                 / (nelem * batch_variance_sqrt * w)
        grad_weight = grad_bias = None
        if gamma is not None:
            grad_weight = ((out_y - b)/w * grad_from_upstream).sum(dim=ax)
        if beta is not None:
            grad_bias = grad_from_upstream.sum(dim=ax)

        return grad_x, grad_weight, grad_bias


class MySigmoid(Function):
    """
    Implement our own custom autograd Functions of Sigmoid by subclassing
    torch.autograd.Function and override the forward and backward passes
    """
    @staticmethod
    def forward(ctx, in_x):
        """ override the forward function """
        sig = 1.0 / (1.0 + torch.exp(-in_x))
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """ override backward function f """
        print('Performing custom backward of MySigmoid')
        sig, = ctx.saved_tensors
        grad_x = sig * (1.0 - sig) * grad_from_upstream
        return grad_x


class MyMSELoss(Function):
    """
    Implement our own custom autograd Functions of MSELoss by subclassing
    torch.autograd.Function and override the forward and backward passes
    """
    @staticmethod
    def forward(ctx, in_x, target):
        """ override the forward function """
        error = in_x - target
        ctx.save_for_backward(error)
        return (error ** 2).mean()

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """ override backward function f """
        print('Performing custom backward of MyMSELoss')
        error, = ctx.saved_tensors
        grad_x = error * 2.0 / error.numel()
        return grad_x, None


# Here is the test of the MyBatchNormNd, MySigmoid and MyMSE
if flag_custom_implement_torch:
    x1 = input_x.detach().clone()
    x1.requires_grad = True
    weight1 = input_weight.detach().clone()
    weight1.requires_grad = True
    bias1 = input_bias.detach().clone()
    bias1.requires_grad = True

    # forward pass
    # apply custom BatchNorm
    y1 = MyBatchNormNd.apply(x1, weight1, bias1)
    y1.retain_grad()
    # apply custom Sigmoid
    out1 = MySigmoid.apply(y1)
    out1.retain_grad()

    # Calculate MSE loss
    loss_mse1 = MyMSELoss.apply(out1, targets)

    # backward pass including custom backward functions of MyBatchNormNd and MySigmoid
    loss_mse1.backward()

'''
Custom implementation #B: implementation of the forward and backward passes 
of BatchNorm-sigmoid-MSE using Numpy
'''
if flag_custom_implement_numpy:
    # get the same inputs, targets and parameters as above
    np_t = targets.detach().numpy()
    np_x = input_x.detach().numpy()
    w_b_shape = [1, -1] + [1] * (np_x.ndim - 2)
    np_w = input_weight.detach().numpy().reshape(w_b_shape)
    np_b = input_bias.detach().numpy().reshape(w_b_shape)

    # forward pass
    axis = tuple([0]) + tuple(range(2, np_x.ndim))
    np_batch_mean = np_x.mean(axis=axis, keepdims=True)
    np_batch_variance_sqrt = np.sqrt(np_x.var(axis, keepdims=True) + eps)
    np_y = (np_x - np_batch_mean) / np_batch_variance_sqrt * np_w + np_b
    np_out = 1.0 / (1.0 + np.exp(-np_y))

    # calculate MSE loss
    loss_mse_np = ((np_out-np_t)**2).mean()

    # backward pass for gradient calculation
    grad_sig_np = (np_out-np_t) * 2.0 / np_out.size
    grad_y_np = np_out * (1.0 - np_out) * grad_sig_np
    n = np_out.size // np_out.shape[1]  # exclude the dimension of channel
    grad_x_np = np_w * grad_y_np / np_batch_variance_sqrt  \
             - np_w * grad_y_np.sum(axis=axis, keepdims=True) / (n * np_batch_variance_sqrt) \
             - (np_y-np_b) * (grad_y_np * (np_y-np_b)).sum(axis=axis, keepdims=True) \
                / (n*np_batch_variance_sqrt*np_w)
    grad_w_np = ((np_y - np_b)/np_w * grad_y_np).sum(axis=axis)
    grad_b_np = grad_y_np.sum(axis=axis)


'''
Reference Test: using the built-in torch.nn.BatchNormNd, torch.Sigmoid and torch.nn.MSELoss
'''
if flag_torch_built_ins_as_ref:
    # get the same inputs, targets and parameters as above
    x3 = input_x.detach().clone()
    x3.requires_grad = True
    # using torch.nn.BatchNorm2d as default
    bn = nn.BatchNorm2d(inCh, affine=True, track_running_stats=False, momentum=1.0)
    if x3.dim() - 2 == 3:
        bn = nn.BatchNorm3d(inCh, affine=True, track_running_stats=False, momentum=1.0)
    elif x3.dim() - 2 == 1:
        bn = nn.BatchNorm1d(inCh, affine=True, track_running_stats=False, momentum=1.0)
    # set the trainable parameters of the BatchNorm module to the same weight and bias as used above
    bn.weight = Parameter(input_weight)
    bn.bias = Parameter(input_bias)

    # forward propagation
    y3 = bn(x3)
    y3.retain_grad()
    out3 = torch.sigmoid(y3)
    out3.retain_grad()

    # Calculate MSE loss
    loss_mse3 = nn.MSELoss(reduction='mean')(out3, targets)

    # backward propagation
    loss_mse3.backward()

# Compare custom implementation using tensors (Test #1) to the reference (test #3)
diff_out_1_3 = out1 - out3
diff_loss_1_3 = loss_mse1 - loss_mse3
diff_grad_x_1_3 = x1.grad - x3.grad
diff_grad_w_1_3 = weight1.grad - bn.weight.grad
diff_grad_b_1_3 = bias1.grad - bn.bias.grad

# clean up the infinite small number due to floating point error
diff_out_1_3[diff_out_1_3 < 1e-15] = 0.0
diff_grad_x_1_3[diff_grad_x_1_3 < 1e-15] = 0.0
diff_grad_w_1_3[diff_grad_w_1_3 < 1e-15] = 0.0
diff_grad_b_1_3[diff_grad_b_1_3 < 1e-15] = 0.0

print('\nDifference between Torch Custom implementation and Torch built-in Reference')
print('diff_out_1_3 max difference:', diff_out_1_3.abs().max().detach().numpy())
print('diff_loss_1_3:', diff_loss_1_3.detach().numpy())
print('diff_grad_x_1_3 max difference:', diff_grad_x_1_3.abs().max().detach().numpy())
print('diff_grad_w_1_3:', diff_grad_w_1_3.detach().numpy())
print('diff_grad_b_1_3:', diff_grad_b_1_3.detach().numpy())

# Compare custom implementation using numpy (Test #2) to the reference (test #3)
diff_out_2_3 = np_out - out3.detach().numpy()
diff_loss_2_3 = loss_mse_np - loss_mse3.detach().numpy()
diff_grad_x_2_3 = grad_x_np - x3.grad.detach().numpy()
diff_grad_w_2_3 = grad_w_np - bn.weight.grad.detach().numpy()
diff_grad_b_2_3 = grad_b_np - bn.bias.grad.detach().numpy()

# clean up those super small numbers due to floating point error
diff_out_2_3[diff_out_2_3 < 1e-15] = 0.0
if diff_loss_2_3 < 1e-15:
    diff_loss_2_3 = 0.0
diff_grad_x_2_3[diff_grad_x_2_3 < 1e-15] = 0.0
diff_grad_w_2_3[diff_grad_w_2_3 < 1e-15] = 0.0
diff_grad_b_2_3[diff_grad_b_2_3 < 1e-15] = 0.0

print('\nDifference between Numpy Custom implementation and Torch built-in Reference')
print('diff_out_2_3 max difference:', np.max(np.abs(diff_out_2_3)))
print('diff_loss_2_3:', diff_loss_2_3)
print('diff_grad_x_2_3 max difference:', np.max(np.abs(diff_grad_x_2_3)))
print('diff_grad_w_2_3:', diff_grad_w_2_3)
print('diff_grad_b_2_3:', diff_grad_b_2_3)
