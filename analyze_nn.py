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

def load_net(name):
    if name == 'resnet18':
        net=resnet.resnet18(pretrained=True)
    else:
        raise NotImplementedError
    net=net.cuda()
    return net

def load_datasets(name):
    if name=='imagenet':
        g=datasets.ImageNetLoaderGenerator('/datasets/imagenet','imagenet',128,8,4)
        test_loader=g.test_loader(shuffle=True)
        calib_loader=g.train_loader()
        calib_loader.dataset.transform=g.transform_test
    else:
        raise NotImplementedError
    return test_loader,calib_loader

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('net_name',type=str)
    parser.add_argument('dataset',type=str)
    parser.add_argument('--statistic_size',type=int,default=32)
    args=parser.parse_args()
    return args

def wrap_modules_in_net(net):
    wrapped_modules={}
    slice_size=4
    act_bits=4
    weight_bits=4
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
            _m.activation_quant_mode='in_quant_unsigned'
            wrapped_modules[name]=_m
            _m.weight.data=m.weight.data
            _m.bias=m.bias
            m.forward_backup=m.forward
            m.forward=_m.forward
            _m.mode='raw'
    return wrapped_modules

def quant_calib(net,wrapped_modules,calib_loader):
    calib_size=512
    for name,module in wrapped_modules.items():
        module.mode='calibration_statistic'
        print(f"calibrate {name}")
        cnt=0
        with torch.no_grad():
            for inp,target in calib_loader:
                inp=inp.cuda()
                net(inp)
                cnt+=inp.size(0)
                if cnt>calib_size:
                    break
        module.calibrate()
        module.mode='quant_forward'

if __name__=='__main__':
    args=parse_args()
    net=load_net(args.net_name)
    test_loader,calib_loader=load_datasets(args.dataset)
    wrapped_modules=wrap_modules_in_net(net)
    quant_calib(net,wrapped_modules,calib_loader)


    statistic_size=args.statistics_size
    for name,module in wrapped_modules.items():
        module.mode='statistic_forward'
    cnt=0
    with torch.no_grad():
        for inp,target in test_loader:
            inp=inp.cuda()
            net(inp)
            cnt+=inp.size(0)
            if cnt>statistic_size:
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

