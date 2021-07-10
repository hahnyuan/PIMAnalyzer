import torch
import torch.nn as nn
import sys
import torchvision.models.resnet as resnet
import datasets
import argparse
import nn_layers.conv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import argparse
from quantization import quantizer

def load_net(name):
    if name == 'resnet18':
        net=resnet.resnet18(pretrained=True)
    else:
        raise NotImplementedError
    net=net.cuda()
    return net

def load_datasets(name,data_root,calib_size=128):
    if name=='imagenet':
        g=datasets.ImageNetLoaderGenerator(data_root,'imagenet',calib_size,128,4)
        test_loader=g.test_loader(shuffle=True)
        calib_loader=g.calib_loader(calib_size)
    elif name=='cifar10':
        g=datasets.CIFARLoaderGenerator(data_root,'cifar10',calib_size,128,4)
        test_loader=g.test_loader(shuffle=False)
        calib_loader=g.calib_loader(calib_size)
    else:
        raise NotImplementedError
    return test_loader,calib_loader

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('net_name',type=str)
    parser.add_argument('dataset',type=str)
    parser.add_argument('--data_root',default="data",type=str)
    parser.add_argument('--statistic_size',type=int,default=32)
    args=parser.parse_args()
    return args

def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias

def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b.data)
    else:
        conv_module.bias.data = b.data
    conv_module.weight.data = w.data

def wrap_modules_in_net(net,act_bits=4,weight_bits=4,fuse_bn=False,layer_quantizer=quantizer.ACIQ):
    wrapped_modules={}
    slice_size=4
    
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d):
            if m.bias is None:
                bias_flag=False
            else:
                bias_flag=True
            _m=nn_layers.conv.BitwiseStatisticConv2d(m.in_channels,m.out_channels,m.kernel_size,m.stride,m.padding,m.dilation,m.groups,bias_flag,m.padding_mode)
            _m.quantize=True
            _m.act_bits=act_bits
            _m.weight_bits=weight_bits
            _m.slice_size=slice_size
            _m.quantizer=layer_quantizer(weight_bits,act_bits)
            wrapped_modules[name]=_m
            _m.weight.data=m.weight.data
            _m.bias=m.bias
            m.forward_backup=m.forward
            m.forward=_m.forward
            _m.mode='raw'
        if isinstance(m,nn.BatchNorm2d):
            # print(wrapped_modules)
            print(f"fuse {name}")
            conv=wrapped_modules[name.replace('bn','conv').replace('downsample.1','downsample.0')]
            conv.bn_fused=True
            fold_bn_into_conv(conv,m)
            # conv.weight.data=w
            # conv.bias=nn.Parameter(b)
            m.forward_back=m.forward
            m.forward=lambda x:x
    return wrapped_modules

def quant_calib(net,wrapped_modules,calib_loader):
    calib_layers=[]
    n_calibration_steps=1
    for name,module in wrapped_modules.items():
        module.mode='calibration_forward'
        calib_layers.append(name)
        n_calibration_steps=max(n_calibration_steps,module.quantizer.n_calibration_steps)
    print(f"prepare calibration for {calib_layers}\n n_calibration_steps={n_calibration_steps}")
    for step in range(n_calibration_steps):
        print(f"Start calibration step={step+1}")
        for name,module in wrapped_modules.items():
            module.quantizer.calibration_step=step+1
        with torch.no_grad():
            for inp,target in calib_loader:
                inp=inp.cuda()
                net(inp)
    for name,module in wrapped_modules.items():
        module.mode='quant_forward'
    print("calibration finished")

if __name__=='__main__':
    args=parse_args()
    net=load_net(args.net_name)
    test_loader,calib_loader=load_datasets(args.dataset,args.data_root)
    wrapped_modules=wrap_modules_in_net(net)
    quant_calib(net,wrapped_modules,calib_loader)

    statistic_size=args.statistic_size
    for name,module in wrapped_modules.items():
        module.mode='statistic_forward'
    cnt=0
    with torch.no_grad():
        for inp,target in test_loader:
            inp=inp.cuda()
            net(inp)
            cnt+=inp.size(0)
            if cnt>=statistic_size:
                break
    
    zero_out_exclude_in_zero_tot=0
    tot_out_exclude_in_zero=0

    tot_zero_in=0
    tot_in=0

    tot_zero_out=0
    tot_out=0

    for name,module in wrapped_modules.items():
        # print(module.statistic)
        s=module.statistic
        in_zero_frac=[s[f'zero_in_{i}']/s['in_num'] for i in range(module.act_bits)]
        print(f"{name} in_zero/tot {in_zero_frac}")
        tot_zero_in+=np.sum([s[f'zero_in_{i}'] for i in range(module.act_bits)])
        tot_in+=np.sum([s['in_num'] for i in range(module.act_bits)])
        


        out_zero_frac=[s[f'zero_out_{i}']/s['out_num'] for i in range(module.weight_bits)]
        print(f"{name} out_zero/tot {out_zero_frac}")
        tot_zero_out+=np.sum([s[f'zero_out_{i}'] for i in range(module.act_bits)])
        tot_out+=np.sum([s[f'out_num'] for i in range(module.act_bits)])
        

        out_zero_frac_exclude_in_zero=[s[f'zero_out_{i}_exclude_in_zero']/(s[f'tot_out_{i}_exclude_in_zero']) for i in range(module.weight_bits)]
        print(f"{name} out_zero_exclude_in_zero/tot_exclude_in_zero {out_zero_frac_exclude_in_zero}")
        zero_out_exclude_in_zero_tot+=np.sum([s[f'zero_out_{i}_exclude_in_zero']/1 for i in range(module.weight_bits)])
        tot_out_exclude_in_zero+=np.sum([s[f'tot_out_{i}_exclude_in_zero'] for i in range(module.weight_bits)])

    print("zero_out_exclude_in_zero_tot/tot_out_exclude_in_zero",zero_out_exclude_in_zero_tot/tot_out_exclude_in_zero)
    print("tot_zero_in/tot_in",tot_zero_in/tot_in)
    print("tot_zero_out/tot_out",tot_zero_out/tot_out)

