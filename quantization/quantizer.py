import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization.quant_functions import *
import numpy as np
from tqdm import tqdm


class ACIQ():
    """
    Implementation of Post training 4-bit quantization of convolutional networks for rapid-deployment NIPS2019 
    """
    def __init__(self,w_bit,a_bit,channel_wise=False,bias_correction=False,online_clip=False) -> None:
        self.w_bit=w_bit
        self.a_bit=a_bit
        if channel_wise:
            raise NotImplementedError
        self.channel_wise=channel_wise
        if bias_correction:
            raise NotImplementedError
        self.bias_correction=bias_correction
        self.online_clip=online_clip
        self.laplace_b=None
        self.calibrated=False
    
    def quant_weight(self,weight):
        if not self.channel_wise:
            with torch.no_grad():
                max=weight.data.abs().max()
                interval=max/(2**(self.w_bit-1)-0.5) # symmetric quantization
                w_int=torch.round_(weight/interval)
                w_sim=w_int*interval
                # bias-correction
                if self.bias_correction:
                    pass
        return w_sim

    def quant_weight_bias(self,weight,bias):
        w_sim=self.quant_weight(weight)
        return w_sim,bias


    def calc_laplace_b(self,tensor):
        if not self.channel_wise:
            laplace_b=((tensor-tensor.mean()).abs()).mean()
            print(f"Debug: laplace_b={laplace_b}")
        return laplace_b

    def get_optimal_clipping_value(self,laplace_b,bitwidth):
        d={2:2.38,3:3.89,4:5.03,5:6.20476633,6:7.41312621,7:8.64561998,8:9.89675977}
        
        if bitwidth in d:
            return d[bitwidth]*laplace_b
        else:
            from scipy.optimize import fsolve
            if not self.channel_wise:
                laplace_b=laplace_b.item()
                def func(alpha):
                    return 2*alpha/(3*2**(2*bitwidth))-2*laplace_b*np.exp(-alpha/laplace_b)
                r=fsolve(func,bitwidth)
                return r


    def quant_activation(self,tensor):
        if self.online_clip:
            laplace_b=self.calc_laplace_b(tensor)
        else:
            laplace_b=self.laplace_b
        alpha=self.get_optimal_clipping_value(laplace_b,self.a_bit)
        interval=alpha/(2**(self.a_bit-1)-0.5) # symmetric quantization
        max_value=2**(self.a_bit-1)
        a_int=torch.round_(tensor/interval).clamp(-max_value,max_value-1)
        a_sim=a_int*interval
        return a_sim
    
    def calibration(self,weight,input):
        self.laplace_b=self.calc_laplace_b(input)
        self.calibrated=True