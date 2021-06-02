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
        self.weight_bits=4
        self.weight_scale=None
        self.act_bits=4
        self.in_scale=None
        self.statistic={'calibration_in':[],'calibration_out':[]}
        
    
    def forward(self, x):
        if self.mode=='raw':
            out=super().forward(x)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_statistic":
            out=self.calibration_statistic(x)
        elif self.mode=='statistic_forward':
            out=self.statistic_forward(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_forward(self,x):
        assert self.weight_scale is not None
        if 'in_quant' in self.activation_quant_mode:
            if self.activation_quant_mode=='in_quant_unsigned':
                in_max_int=2**(self.act_bits)-1
            else:
                in_max_int=2**(self.act_bits-1)-1
            in_integer=torch.round_(x/self.in_scale).clamp_(-in_max_int,in_max_int)
            x=in_integer*self.in_scale
        w_max_int=2**(self.weight_bits)-1
        integer=torch.round_(self.weight.data/self.weight_scale).clamp_(-w_max_int,w_max_int)
        w_q=integer*self.weight_scale
        out_q=F.conv2d(x,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
        return out_q
    
    def calibration_statistic(self,x):
        out=super().forward(x)
        self.statistic['calibration_in'].append(x.detach().cpu())
        self.statistic['calibration_out'].append(out.detach().cpu())
        return out
    
    def calibrate(self,clear_data=True,p=2.4,co_optimize=False):
        assert len(self.statistic['calibration_in'])!=0
        xs=torch.cat(self.statistic['calibration_in'],0).cuda()
        ys=torch.cat(self.statistic['calibration_out'],0).cuda()
        w_absmax=self.weight.data.abs().max()
        pos_intervals=2**(self.weight_bits-1)-1
        max_int=2**(self.weight_bits-1)-1
        if 'in_quant' in self.activation_quant_mode:
            in_absmax=xs.data.abs().max()
            if self.activation_quant_mode=='in_quant_unsigned':
                in_pos_intervals=2**(self.act_bits)-1
                in_max_int=2**(self.act_bits)-1
            else:
                in_pos_intervals=2**(self.act_bits-1)-1
                in_max_int=2**(self.act_bits-1)-1
        
        if co_optimize:
            min_score=1e9
            t=tqdm(range(40))
            for weight_i in t:
                for activation_i in range(40):
                    if 'in_quant' in self.activation_quant_mode:
                        new_in_max = in_absmax * (1.0 - (activation_i * 0.015))
                        in_scale=new_in_max/in_pos_intervals
                        xs_q=torch.round_(xs/in_scale).clamp_(-in_max_int,in_max_int).mul_(in_scale)
                    else:
                        in_scale=None
                        xs_q=xs
                    new_max = w_absmax * (1.0 - (weight_i * 0.015))
                    scale=new_max/pos_intervals
                    w_q=torch.round_(self.weight.data/scale).clamp_(-max_int,max_int).mul_(scale)
                    out_q=F.conv2d(xs_q,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
                    # value_q_after_act=self.reconstruct_act_func(out_q)
                    score = lp_loss(ys, out_q, p=p, reduction='all')
                    if score < min_score:
                        min_score = score
                        best_in_scale=in_scale
                        best_weight_scale=scale
                    del xs_q,w_q,out_q
                t.set_postfix({'min_score':min_score.item(),'best_in_scale':best_in_scale.item(),'best_weight_scale':best_weight_scale.item()})
        else:
            min_score=1e9
            for activation_i in range(40):
                if 'in_quant' in self.activation_quant_mode:
                    new_in_max = in_absmax * (1.0 - (activation_i * 0.015))
                    in_scale=new_in_max/in_pos_intervals
                    xs_q=torch.round_(xs/in_scale).clamp_(-in_max_int,in_max_int).mul_(in_scale)
                else:
                    in_scale=None
                    xs_q=xs
                out_q=F.conv2d(xs_q,self.weight.data,self.bias,self.stride,self.padding,self.dilation,self.groups)
                # value_q_after_act=self.reconstruct_act_func(out_q)
                score = lp_loss(ys, out_q, p=p, reduction='all')
                if score < min_score:
                    min_score = score
                    best_in_scale=in_scale
                del xs_q,out_q
            min_score=1e9
            for weight_i in range(40):
                if 'in_quant' in self.activation_quant_mode:
                    xs_q=torch.round_(xs/best_in_scale).clamp_(-in_max_int,in_max_int).mul_(best_in_scale)
                else:
                    in_scale=None
                    xs_q=xs
                new_max = w_absmax * (1.0 - (weight_i * 0.015))
                scale=new_max/pos_intervals
                w_q=torch.round_(self.weight.data/scale).clamp_(-max_int,max_int).mul_(scale)
                out_q=F.conv2d(xs_q,w_q,self.bias,self.stride,self.padding,self.dilation,self.groups)
                # value_q_after_act=self.reconstruct_act_func(out_q)
                score = lp_loss(ys, out_q, p=p, reduction='all')
                if score < min_score:
                    min_score = score
                    best_weight_scale=scale
                del xs_q,w_q,out_q
        assert best_weight_scale is not None
        self.in_scale=best_in_scale
        self.weight_scale=best_weight_scale
        if clear_data:
            self.statistic['calibration_in'].clear()
            self.statistic['calibration_out'].clear()

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
        
    def statistic_forward(self,x):
        if 'in_quant' in self.activation_quant_mode:
            if self.activation_quant_mode=='in_quant_unsigned':
                in_max_int=2**(self.act_bits)-1
            else:
                in_max_int=2**(self.act_bits-1)-1
            in_integer=torch.round_(x/self.in_scale).clamp_(-in_max_int,in_max_int)
            x=in_integer*self.in_scale
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