import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization.quant_functions import *
import numpy as np
from tqdm import tqdm


class QuantizeConv2d(nn.Conv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        
        self.mode=None
        self.activation_quant_mode=None # in_quant, in_quant_unsigned
        self.weight_bits=None
        self.weight_scale=None
        self.act_bits=None
        self.x_scale=None
    
    def forward(self, x):
        if self.mode=='raw':
            out=super().forward(x)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_forward":
            out=self.calibrate_forward(x)
        elif self.mode=='statistic_forward':
            out=self.statistic_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        assert self.weight_scale is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        if 'in_quant' in self.activation_quant_mode:
            if self.activation_quant_mode=='in_quant_unsigned':
                in_max_int=2**(self.act_bits)-1
            else:
                in_max_int=2**(self.act_bits-1)-1
            in_integer=torch.round_(x/self.x_scale).clamp_(-in_max_int,in_max_int)
            x=in_integer*self.x_scale
        w_max_int=2**(self.weight_bits)-1
        integer=torch.round_(self.weight.data/self.weight_scale).clamp_(-w_max_int,w_max_int)
        w_q=integer*self.weight_scale
        out_q=F.conv2d(x,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
        return out_q
    
    def calibrate_forward(self,x_sim):
        assert self.weight_bits is not None and self.act_bits is not None, f"You should set the weight_bits and bias_bits for {self}"
        if not hasattr(x_sim,'scale'):
            x_sim=SignedQuantizeCalibration().apply(x_sim,self.act_bits)
        # raw_out=super().forward(x_sim)
        weight_sim=SignedQuantizeCalibration().apply(self.weight,self.weight_bits)
        self.x_scale=xscale=x_sim.scale
        self.weight_integer=weight_sim.integer
        self.weight_scale=weight_sim.scale
        if self.bias is not None:
            # bias_sim=SignedQuantizeCalibration().apply(self.bias,16)
            bias_scale=self.x_scale*self.weight_scale
            bias_integer=torch.round(self.bias/bias_scale).clamp(-2**15,2**15-1)
            bias_sim=bias_integer*bias_scale
            self.bias_scale=bias_scale
            self.bias_integer=bias_integer
        else:
            bias_sim=None
        out_sim_=F.conv2d(x_sim, weight_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        out_sim=SignedQuantizeCalibration().apply(out_sim_,self.act_bits)
        return out_sim

class BitwiseStatisticConv2d(QuantizeConv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.slice_size=None
        self.statistic={}
        
    def statistic_forward(self,x):
        if 'in_quant' in self.activation_quant_mode:
            if self.activation_quant_mode=='in_quant_unsigned':
                in_max_int=2**(self.act_bits)-1
            else:
                in_max_int=2**(self.act_bits-1)-1
            in_integer=torch.round_(x/self.x_scale).clamp_(-in_max_int,in_max_int)
            x=in_integer*self.x_scale
        w_max_int=2**(self.weight_bits)-1
        w_integer=torch.round_(self.weight.data/self.weight_scale).clamp_(-w_max_int,w_max_int)
        w_q=w_integer*self.weight_scale
        raw_out=F.conv2d(x,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
        
        b,oc,oh,ow=raw_out.size()
        
        kernel_size=self.weight.size()[2:]
        x_unfolded=F.unfold(in_integer,kernel_size,self.dilation,self.padding,self.stride) # shape N,C×∏(kernel_size),L
        W=w_integer.view(oc,-1) # shape oc,C*∏(kernel_size)
        b,win_size,n_window=x_unfolded.size()
        n_slice=win_size//self.slice_size
        # ignore the un-aligned datas
        x_unfolded=x_unfolded[:,:n_slice*self.slice_size]
        W=W[:,:n_slice*self.slice_size]
        
        all_x_slice=x_unfolded.view(b,n_slice,self.slice_size,n_window)
        
        W_slice=W.view(oc,n_slice,self.slice_size)
        # print(f"x_slice {x_slice.size()} W_slice {W_slice.size()}")
        S=0 # shape N,oc,L
        all_x_slice=all_x_slice.long()
        W_slice=W_slice.long()
        if f'in_num' not in self.statistic:
            self.statistic[f'in_num']=0
        self.statistic['in_num']+=b*n_slice*n_window
        if f'out_num' not in self.statistic:
            self.statistic[f'out_num']=0
        self.statistic['out_num']+=b*n_slice*oc*n_window*self.act_bits
        with torch.no_grad():
            for act_bit_i in range(self.act_bits):
                x_bit=((all_x_slice>>act_bit_i)&1).float()
                zero_in_num=torch.sum((torch.sum(x_bit,2)==0).long()).item()
                if f'zero_in_{act_bit_i}' not in self.statistic:
                    self.statistic[f'zero_in_{act_bit_i}']=0
                self.statistic[f'zero_in_{act_bit_i}']+=zero_in_num
                
                for w_bit_i in range(self.weight_bits):
                    w_bit=((W_slice>>w_bit_i)&1).float()
                    zero_out_num=0
                    for i in range(n_slice):
                        psum=torch.matmul(w_bit[:,i],x_bit[:,i])
                        # if i not in self.statistic:
                        #     self.statistic[i]=[]
                        # psum_sorted=torch.sort(psum.view(-1))[0][::int(psum.view(-1).size(0)/1000)]
                        # self.statistic[i].append(psum_sorted.detach().cpu().numpy())
                        zero_out_num+=torch.sum((psum==0).long()).item()
                    if f'zero_out_{w_bit_i}' not in self.statistic:
                        self.statistic[f'zero_out_{w_bit_i}']=0
                    self.statistic[f'zero_out_{w_bit_i}']+=zero_out_num
                    if f'zero_out_{w_bit_i}_exclude_in_zero' not in self.statistic:
                        self.statistic[f'zero_out_{w_bit_i}_exclude_in_zero']=0
                    # shape of zero out: n_slice*(b*oc*L); shape of zero in: b*n_slice*L
                    self.statistic[f'zero_out_{w_bit_i}_exclude_in_zero']+=zero_out_num-oc*zero_in_num
                    if f'tot_out_{w_bit_i}_exclude_in_zero' not in self.statistic:
                        self.statistic[f'tot_out_{w_bit_i}_exclude_in_zero']=0
                    # shape of zero out: n_slice*(b*oc*L); shape of zero in: b*n_slice*L
                    self.statistic[f'tot_out_{w_bit_i}_exclude_in_zero']+=psum.numel()*n_slice-oc*zero_in_num
                
        return raw_out