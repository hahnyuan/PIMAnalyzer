"""
Reuse version v2
Author: Hahn Yuan
"""

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from quantization.quant_functions import *
import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve


class BaseQuantizer():
    def __init__(self,w_bit,a_bit) -> None:
        self.w_bit=w_bit
        self.a_bit=a_bit
        self.n_calibration_steps=1
        self.calibration_step=1
        self.calibrated=False

    def quant_weight_bias(self,weight,bias):
        return weight,bias

    def quant_activation(self,x):
        return x
    
    def quant_output(self,out):
        return out
    
    def calibration(self,x,weight,bias,op)->Tensor:
        return op(x,weight,bias)

class ACIQ(BaseQuantizer):
    """
    Implementation of Post training 4-bit quantization of convolutional networks for rapid-deployment NIPS2019 
    """
    def __init__(self,w_bit,a_bit,channel_wise=False,bias_correction=False,online_clip=False) -> None:
        super().__init__(w_bit,a_bit)
        self.channel_wise=channel_wise
        if bias_correction:
            raise NotImplementedError
        self.bias_correction=bias_correction
        self.online_clip=online_clip
        self.laplace_b=None
    
    def quant_weight(self,weight):
        with torch.no_grad():
            if not self.channel_wise:
                max=weight.data.abs().max()
            else:
                max=weight.data.abs().view(weight.size(0),-1).max(1)[0].view(-1,*[1]*(weight.dim()-1))
            interval=max/(2**(self.w_bit-1)-0.5) # symmetric quantization, do not need clamp
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
        else:
            tensor=tensor.transpose(0,1).reshape(tensor.size(1),-1)
            laplace_b=((tensor-tensor.mean(1,keepdim=True)).abs()).mean(1)
        print(f"Debug: laplace_b={laplace_b}")
        return laplace_b

    def get_optimal_clipping_value(self,laplace_b=None,bitwidth=None):
        if laplace_b is None:laplace_b=self.laplace_b
        if bitwidth is None:bitwidth=self.a_bit
        d={2:2.38,3:3.89,4:5.03,5:6.20476633,6:7.41312621,7:8.64561998,8:9.89675977}
        if bitwidth in d:
            return d[bitwidth]*laplace_b
        else:
            def func(alpha):
                return 2*alpha/(3*2**(2*bitwidth))-2*np.exp(-alpha)
            r=fsolve(func,bitwidth)
            return r*laplace_b

    def quant_activation(self,tensor):
        if self.online_clip:
            laplace_b=self.calc_laplace_b(tensor)
        else:
            laplace_b=self.laplace_b
        alpha=self.get_optimal_clipping_value(laplace_b,self.a_bit)
        interval=alpha/(2**(self.a_bit-1)-0.5) # symmetric quantization
        if self.channel_wise:
            interval=interval.view(1,-1,*[1]*(tensor.dim()-2))
        max_value=2**(self.a_bit-1)
        a_int=torch.round_(tensor/interval).clamp(-max_value,max_value-1)
        a_sim=a_int*interval
        return a_sim
    
    def calibration(self,x,weight,bias,op):
        laplace_b=self.calc_laplace_b(x)
        if self.laplace_b is None:
            self.laplace_b=laplace_b
        else:
            self.laplace_b=self.laplace_b*0.9+0.1*laplace_b
        self.calibrated=True
        weight_sim,bias_sim=self.quant_weight_bias(weight,bias)
        x_sim=self.quant_activation(x)
        output_sim=op(x_sim, weight_sim, bias_sim)
        return output_sim


class DynamicACIQ(ACIQ):
    def __init__(self,w_bit,a_bit,channel_wise=False,bias_correction=False,online_clip=False,max_interval_up=1,interval_multiplier=2) -> None:
        super().__init__(w_bit, a_bit, channel_wise=channel_wise, bias_correction=bias_correction, online_clip=online_clip)
        self.max_interval_up=max_interval_up
        self.interval_multiplier=interval_multiplier
    
    def quant_activation(self,tensor):
        if self.online_clip:
            laplace_b=self.calc_laplace_b(tensor)
        else:
            laplace_b=self.laplace_b
        alpha=self.get_optimal_clipping_value(laplace_b,self.a_bit)
        interval=alpha/(2**(self.a_bit-1)-0.5) # symmetric quantization
        if self.channel_wise:
            interval=interval.view(1,-1,*[1]*(tensor.dim()-2))
        max_value=2**(self.a_bit-1)
        a_int=torch.round_(tensor/interval)#.clamp(-max_value,max_value-1)
        for i in range(self.max_interval_up):
            if self.channel_wise:
                for c in range(tensor.size(1)):
                    if (a_int[:,c].abs()>max_value).any():
                        interval[:,c]*=self.interval_multiplier
                        a_int[:,c]=torch.round_(tensor[:,c]/interval[:,c])
            else:
                if (a_int.abs()>max_value).any():
                    interval*=self.interval_multiplier
                    a_int=torch.round_(tensor/interval)

        a_sim=a_int.clamp(-max_value,max_value-1)*interval
        return a_sim

class EasyQuant(BaseQuantizer):
    """
    Implementation of EasyQuant: Post-training Quantization via Scale Optimization arxiv2020 
    """
    def __init__(self, w_bit, a_bit,channel_wise=False,eq_alpha=0.5,eq_beta=2,eq_n=100) -> None:
        super().__init__(w_bit, a_bit)
        self.n_calibration_steps=3
        self.raw_outs=[]
        self.channel_wise=channel_wise
        self.eq_alpha=eq_alpha
        self.eq_beta=eq_beta
        self.eq_n=eq_n
        self.weight_interval=None
        self.input_interval=None

    def quant_weight(self,weight,weight_interval=None):
        if weight_interval is None:
            weight_interval=self.weight_interval
        with torch.no_grad():
            if self.channel_wise:
                weight_interval=weight_interval.view(-1,*[1]*(weight.dim()-1))
            max_value=2**(self.w_bit-1)
            w_int=torch.round_(weight/weight_interval).clamp(-max_value,max_value-1)
            w_sim=w_int*weight_interval
            # bias-correction
        return w_sim
    
    def quant_weight_bias(self, weight, bias):
        return self.quant_weight(weight),bias

    def quant_activation(self,tensor,input_interval=None):
        if input_interval is None: input_interval=self.input_interval
        if self.channel_wise:
            input_interval=input_interval.view(1,-1,*[1]*(tensor.dim()-2))
        max_value=2**(self.a_bit-1)
        a_int=torch.round_(tensor/input_interval).clamp(-max_value,max_value-1)
        a_sim=a_int*input_interval
        return a_sim

    def search_best_weight(self,x,weight,bias,op,raw_out,init_interval):
        max_similarity=-2
        best_weight_interval=None
        best_out=None
        for i in range(self.eq_n):
            now_interval=(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval
            max_value=2**(self.w_bit-1)
            w_int=torch.round_(weight/now_interval).clamp(-max_value,max_value-1)
            w_sim=w_int*now_interval
            output_sim=op(x,w_sim,bias)
            # TODO: bias quantization
            similarity=F.cosine_similarity(output_sim.view(-1),raw_out.view(-1),0)
            if similarity>max_similarity:
                best_weight_interval=now_interval
                max_similarity=similarity
                best_out=output_sim
        return best_weight_interval,best_out

    def search_best_input(self,x,w_sim,b_sim,op,raw_out,init_interval):
        max_similarity=-2
        best_input_interval=None
        best_out=None
        for i in range(self.eq_n):
            now_interval=(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval
            max_value=2**(self.a_bit-1)
            a_int=torch.round_(x/now_interval).clamp(-max_value,max_value-1)
            a_sim=a_int*now_interval
            output_sim=op(a_sim,w_sim,b_sim)
            similarity=F.cosine_similarity(output_sim.view(-1),raw_out.view(-1),0)
            if similarity>max_similarity:
                best_input_interval=now_interval
                max_similarity=similarity
                best_out=output_sim
        return best_input_interval,best_out

    def calibration(self,x,weight,bias,op):
        # step1: collection the FP32 values
        if self.calibration_step==1:
            out=op(x,weight,bias)
            self.raw_outs.append(out.cpu().detach())
            return out
        # step1: search for the best S^w of each layer
        elif self.calibration_step==2:
            # initialize
            if self.channel_wise:
                max=weight.data.abs().view(weight.size(0),-1).max(1)[0]
                max=max.view(-1,*[1]*(weight.dim()-1))
            else:
                max=weight.data.abs().max()
            init_interval=max/(2**(self.w_bit-1)-0.5) # symmetric quantization
            raw_out=torch.cat(self.raw_outs,0).to(x.device)
            self.weight_interval,best_out=self.search_best_weight(x,weight,bias,op,raw_out,init_interval)
            print(f"Set weight_interval={self.weight_interval.view(-1)}")
            return best_out
        # step3: search for the best S^a of each layer
        elif self.calibration_step==3:
            w_sim,b_sim=self.quant_weight_bias(weight,bias)
            # initialize
            if self.channel_wise:
                max=x.data.abs().transpose(0,1).reshape(x.size(1),-1).max(1)[0]
                max=max.view(1,-1,*[1]*(weight.dim()-2))
            else:
                max=x.data.abs().max()
            init_interval=max/(2**(self.a_bit-1)-0.5) # symmetric quantization
            raw_out=torch.cat(self.raw_outs,0).to(x.device)
            self.input_interval,best_out=self.search_best_input(x,w_sim,b_sim,op,raw_out,init_interval)
            print(f"Set input_interval={self.input_interval.view(-1)}")
            return best_out

class DirectPowerOf2EasyQuant(EasyQuant):
    def __init__(self, w_bit, a_bit,channel_wise=False,eq_alpha=0.5,eq_beta=2,eq_n=10) -> None:
        super().__init__(w_bit, a_bit, channel_wise=channel_wise, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n)

    def search_best_weight(self,x,weight,bias,op,raw_out,init_interval):
        init_interval=2**torch.round(torch.log2(init_interval))
        max_similarity=-2
        best_weight_interval=None
        best_out=None
        for i in range(-self.eq_n//2,self.eq_n//2):
            now_interval=2**i*init_interval
            max_value=2**(self.w_bit-1)
            w_int=torch.round_(weight/now_interval).clamp(-max_value,max_value-1)
            w_sim=w_int*now_interval
            output_sim=op(x,w_sim,bias)
            # TODO: bias quantization
            similarity=F.cosine_similarity(output_sim.view(-1),raw_out.view(-1),0)
            if similarity>max_similarity:
                best_weight_interval=now_interval
                max_similarity=similarity
                best_out=output_sim
        return best_weight_interval,best_out

    def search_best_input(self,x,w_sim,b_sim,op,raw_out,init_interval):
        init_interval=2**torch.round(torch.log2(init_interval))
        max_similarity=-2
        best_input_interval=None
        best_out=None
        for i in range(-self.eq_n//2,self.eq_n//2):
            now_interval=2**i*init_interval
            now_interval=(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval
            max_value=2**(self.a_bit-1)
            a_int=torch.round_(x/now_interval).clamp(-max_value,max_value-1)
            a_sim=a_int*now_interval
            output_sim=op(a_sim,w_sim,b_sim)
            similarity=F.cosine_similarity(output_sim.view(-1),raw_out.view(-1),0)
            if similarity>max_similarity:
                best_input_interval=now_interval
                max_similarity=similarity
                best_out=output_sim
        return best_input_interval,best_out

class PowerOf2EasyQuant(EasyQuant):
    def __init__(self, w_bit, a_bit,channel_wise=False,eq_alpha=0.5,eq_beta=2,eq_n=100) -> None:
        super().__init__(w_bit, a_bit, channel_wise=channel_wise, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n)
        self.output_interval=None
    
    def quant_output(self,output,output_interval=None):
        if output_interval is None: output_interval=self.output_interval
        if self.channel_wise:
            output_interval=output_interval.view(1,-1,*[1]*(output.dim()-2))
        max_value=2**(self.a_bit-1)
        a_int=torch.round_(output/output_interval).clamp(-max_value,max_value-1)
        a_sim=a_int*output_interval
        return a_sim

    def search_best_input_output(self,x,w_sim,b_sim,op,raw_out,init_interval):
        max_similarity=-2
        best_input_interval=None
        best_output_interval=None
        best_out=None
        for i in range(self.eq_n):
            now_interval=(self.eq_alpha+i/self.eq_n*(self.eq_beta-self.eq_alpha))*init_interval
            max_value=2**(self.a_bit-1)
            a_int=torch.round_(x/now_interval).clamp(-max_value,max_value-1)
            a_sim=a_int*now_interval
            output_sim=op(a_sim,w_sim,b_sim)
            init_output_interval=output_sim.max()
            if self.channel_wise:
                raise NotImplementedError
            init_output_interval/=(2**(self.a_bit-1)-0.5)
            tmp_scale=self.weight_interval*now_interval
            init_output_interval=tmp_scale/2**torch.round(torch.log2(tmp_scale/init_output_interval))
            for j in range(-4,4):
                now_output_interval=2**j*init_output_interval
                output_q_int=torch.round_(output_sim/now_output_interval).clamp(-max_value,max_value-1)
                output_q_sim=output_q_int*now_output_interval
                similarity=F.cosine_similarity(output_q_sim.view(-1),raw_out.view(-1),0)
                if similarity>max_similarity:
                    best_input_interval=now_interval
                    best_output_interval=now_output_interval
                    max_similarity=similarity
                    best_out=output_sim
        return best_input_interval,best_output_interval,best_out
    
    def calibration(self,x,weight,bias,op):
        # step1: collection the FP32 values
        if self.calibration_step==1:
            out=op(x,weight,bias)
            self.raw_outs.append(out.cpu().detach())
            return out
        # step1: search for the best S^w of each layer
        elif self.calibration_step==2:
            # initialize
            if self.channel_wise:
                max=weight.data.abs().view(weight.size(0),-1).max(1)[0]
                max=max.view(-1,*[1]*(weight.dim()-1))
            else:
                max=weight.data.abs().max()
            init_interval=max/(2**(self.w_bit-1)-0.5) # symmetric quantization
            raw_out=torch.cat(self.raw_outs,0).to(x.device)
            self.weight_interval,best_out=self.search_best_weight(x,weight,bias,op,raw_out,init_interval)
            print(f"Set weight_interval={self.weight_interval.view(-1)}")
            return best_out
        # step3: search for the best S^a of each layer
        elif self.calibration_step==3:
            w_sim,b_sim=self.quant_weight_bias(weight,bias)
            # initialize
            if self.channel_wise:
                max=x.data.abs().transpose(0,1).reshape(x.size(1),-1).max(1)[0]
                max=max.view(1,-1,*[1]*(weight.dim()-2))
            else:
                max=x.data.abs().max()
            init_interval=max/(2**(self.a_bit-1)-0.5) # symmetric quantization
            raw_out=torch.cat(self.raw_outs,0).to(x.device)
            self.input_interval,self.output_interval,best_out=self.search_best_input_output(x,w_sim,b_sim,op,raw_out,init_interval)
            print(f"Set input_interval={self.input_interval.view(-1)} output_interval={self.output_interval.view(-1)}")
            return best_out