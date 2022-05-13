from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])



def get_scale_zero_point(min_val: float, max_val: float ,num_bits=8):
    
    """
    Given a tensor, calculates scale and zero point for a tensor to be 
    quantized
    """
    # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale
    
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x: torch.tensor, num_bits=8, min_val=None, max_val=None):
    
    """
    Quantizes a tensor, returns a named tuple in pytorch Qtensor format
    """
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = get_scale_zero_point(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
   
    """
    Dequantizes a tensor given a Qtensor
    """
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)



# class FakeQuantOp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
#         x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
#         x = dequantize_tensor(x)
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         # straight through estimator
#         return grad_output, None, None, None





def simulate_quantization(net: nn.Module, stats: dict):
    
    """
        This function tranverses through the layers of a network, 
        stores its orignal weights for backward pass and 
        quantizes-dequantizes its weights. This also populate a running stats 
        dictionary which behaves like a pytorch observer class.  
    """

    original_weights = {}

    for layer in net.children():
        layer_name = layer._get_name()
        for key in layer._parameters.keys():
            name = layer_name + '.' + key
            # hold onto original weights 
            weights = layer._parameters[key].data
            original_weights[name] = weights
            # receives a named tuple 
            weights  = quantize_tensor(weights)
            stats[name] = tuple((weights.scale, weights.zero_point))

            # dequantization happens on quantized weights 
            # they have some precision loss from original weights 
            weights = dequantize_tensor(weights)

            # replace weights with dequantized weights
            layer._parameters[key].data = weights
    
    return original_weights


def restore_original_weights(net: nn.Module, weights: dict):
    """
    Restore the original weights of the network for the backward 
    pass.
    """
    for layer in net.children():
        layer_name = layer._get_name()
        for key in layer._parameters.keys():
            name = layer_name + '.' + key
            layer._parameters[key].data = weights[name] 