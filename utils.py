import torch
import torch.nn.functional as F
from torch import autograd

def dequantize(x):
    '''Dequantize data.
    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).
    Args:
        x: input tensor.
    Returns:
        dequantized data.
    '''
    noise = torch.distributions.Uniform(0., 1.).sample(x.size())
    return (x * 255. + noise) / 256.

def prepare_data(x):
    assert len(list(x.size())) == 4
    [B, C, H, W] = list(x.size())
    x = dequantize(x)
    return x

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def regularization_term(d_out,x_in):
    reg = compute_grad2(d_out, x_in).mean()
    return reg