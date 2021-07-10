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
        self.quantizer=None
        self.weight_bits=None
        self.weight_scale=None
        self.act_bits=None
        self.x_scale=None
        self.bn_fused=False
    
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
        # assert self.weight_scale is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        assert self.quantizer.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        # if 'in_quant' in self.activation_quant_mode:
        #     if self.activation_quant_mode=='in_quant_unsigned':
        #         in_max_int=2**(self.act_bits)-1
        #     else:
        #         in_max_int=2**(self.act_bits-1)-1
        #     in_integer=torch.round_(x/self.x_scale).clamp_(-in_max_int,in_max_int)
        #     x=in_integer*self.x_scale
        # w_max_int=2**(self.weight_bits)-1
        # integer=torch.round_(self.weight.data/self.weight_scale).clamp_(-w_max_int,w_max_int)
        # w_q=integer*self.weight_scale
        # out_q=F.conv2d(x,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
        weight_sim,bias_sim=self.quantizer.quant_weight_bias(self.weight,self.bias)
        x_sim=self.quantizer.quant_activation(x)
        out_sim=F.conv2d(x_sim, weight_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        out_sim=self.quantizer.quant_output(out_sim)
        return out_sim
    
    def get_quant_weight_bias(self):
        return self.quantizer.quant_weight_bias(self.weight,self.bias)

    def calibrate_forward(self,x):
        assert self.weight_bits is not None and self.act_bits is not None, f"You should set the weight_bits and bias_bits for {self}"
        op=lambda input,weight,bias:F.conv2d(input,weight,bias,self.stride,self.padding, self.dilation, self.groups)
        out_sim=self.quantizer.calibration(x,self.weight,self.bias,op)
        # weight_sim,bias_sim=self.quantizer.quant_weight_bias(self.weight,self.bias)
        # x_sim=self.quantizer.quant_activation(x)
        # if not hasattr(x_sim,'scale'):
        #     x_sim=SignedQuantizeCalibration().apply(x_sim,self.act_bits)
        # # raw_out=super().forward(x_sim)
        # weight_sim=SignedQuantizeCalibration().apply(self.weight,self.weight_bits)
        # self.x_scale=xscale=x_sim.scale
        # self.weight_integer=weight_sim.integer
        # self.weight_scale=weight_sim.scale
        # if self.bias is not None:
        #     # bias_sim=SignedQuantizeCalibration().apply(self.bias,16)
        #     bias_scale=self.x_scale*self.weight_scale
        #     bias_integer=torch.round(self.bias/bias_scale).clamp(-2**15,2**15-1)
        #     bias_sim=bias_integer*bias_scale
        #     self.bias_scale=bias_scale
        #     self.bias_integer=bias_integer
        # else:
        #     bias_sim=None
        # out_sim=F.conv2d(x_sim, weight_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        # out_sim=SignedQuantizeCalibration().apply(out_sim_,self.act_bits)
        return out_sim

class StatisticConv2d(QuantizeConv2d):
    def __init__(self,in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.slice_size=None
        self.statistic={'crossbar_weight':[]}
    
    def statistic_forward(self,x):
        raw_w=self.weight.data
        n_slices=self.slice_size
        self.statistic[f'crossbar_weight']=raw_w.view(self.out_channels,self.slice_size)
        # print(f"x_slice {x_slice.size()} W_slice {W_slice.size()}")
        return raw_out


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

class BaseMappedConv2d(nn.Conv2d):
    """
    Map the nn.Conv2d to different size of crossbar, the MVM is processed in floating point
    """
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
        
        self.mode='raw`'

        self.crossbar_cols=None
        self.crossbar_rows=None
        self.n_cell_per_weight=None
        self.n_input_steps=None
    
    def map_to_crossbars(self,rows,cols,n_cell_per_weight=1,n_input_steps=1):
        self.crossbar_cols=cols
        self.crossbar_rows=rows
        self.n_cell_per_weight=n_cell_per_weight
        self.n_input_steps=n_input_steps
        self.crossbars=nn.ModuleList()
        assert n_cell_per_weight==1,f"BaseMappedConv2d only support n_cell_per_weight=1"
        assert n_input_steps==1,f"BaseMappedConv2d only support n_input_steps=1"
        input_size=self.in_channels//self.groups*self.kernel_size[0]*self.kernel_size[1]
        out_c_ind=0
        while out_c_ind<self.out_channels:
            w_col_chunk=self.weight[out_c_ind:out_c_ind+self.crossbar_cols]
            input_ind=0
            while input_ind<input_size:
                pass
            out_c_ind+=self.crossbar_cols


    def forward(self, x):
        if self.mode=='raw':
            out=super().forward(x)
        elif self.mode=="forward":
            out=self.mapped_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def mapped_forward(self,x):
        assert self.crossbar_cols is not None,f"You should map the conv to hte crossbar before using mapped_forward"
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

class BitwiseQuantMappedConv2d(QuantizeConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride, padding, dilation, groups: int, bias: bool, padding_mode: str):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)