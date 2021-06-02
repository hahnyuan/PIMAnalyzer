import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function
import numpy as np



def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs_().pow_(p).sum(1).mean()
    else:
        return (pred-tgt).abs_().pow_(p).mean()

def search_best_scale_mse(x,pos_intervals,max_int):
    x_absmax = x.abs().max()
    best_score = 1e6
    best_scale=None
    best_max_frac=None
    for i in range(80):
        new_max = x_absmax * (1.0 - (i * 0.01))
        # x_q = self.quantize(x, new_max)
        scale=new_max/pos_intervals
        integer=torch.round_(x/scale).clamp_(-max_int,max_int)
        x_q=integer*scale
        # L_p norm minimization as described in LAPQ
        # https://arxiv.org/abs/1911.07190
        score = lp_loss(x, x_q, p=2.4, reduction='all')
        if score < best_score:
            best_score = score
            best_scale=scale
            # best_max_frac=(1.0 - (i * 0.01))
    # print(f"best_max_frac {best_max_frac}")
    return best_scale

class UnsignedQuantizeCalibration(Function):
    """
    Unsigned Quantization.
    Scale is defined by |x|_max.
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input,bit_width):
        # ctx.save_for_backward(input)
        intervals=2**bit_width-1
        max_int=2**bit_width-1
        # quantile
        in_sort=input.view(-1).sort()[0]
        in_max=in_sort[int(len(in_sort)*0.995)]
        # in_max=input.max()
        scale=in_max/intervals
        integer=torch.round_(input/scale).clamp_(0,max_int)
        out=integer*scale
        out.integer=integer
        out.scale=scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class UnsignedQuantize(Function):
    """
    Unsigned Quantization.
    Scale is defined by |x|_max.
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input ,bit_width,scale):
        # ctx.save_for_backward(input)
        max_int=2**bit_width-1
        integer=torch.round_(input/scale).clamp_(0,max_int)
        out=integer*scale
        out.integer=integer
        out.scale=scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class SignedQuantizeCalibration(UnsignedQuantize):
    """
    Signed Quantization.
    Scale is defined by |x|_max.
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input,bit_width):
        # ctx.save_for_backward(input)
        pos_intervals=2**(bit_width-1)-1
        max_int=2**(bit_width-1)-1
        scale=search_best_scale_mse(input,pos_intervals,max_int)
        integer=torch.round_(input/scale).clamp_(-max_int,max_int)
        out=integer*scale
        out.integer=integer
        out.scale=scale
        return out

class SignedQuantize(UnsignedQuantize):
    """
    Signed Quantization.
    Scale is defined by |x|_max.
    Backward using STE. 
    """
    @staticmethod
    def forward(ctx, input ,bit_width,scale):
        # ctx.save_for_backward(input)
        max_int=2**(bit_width-1)-1
        integer=torch.round_(input/scale).clamp_(-max_int,max_int)
        out=integer*scale
        out.integer=integer
        out.scale=scale
        return out

class Identity:
    def __call__(self,*args):
        return args
    