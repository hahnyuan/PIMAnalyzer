import torch
import torch.nn as nn


class CrossBar(nn.Module):
    def __init__(self,rows,cols,open_rows_per_run=None):
        super().__init__()
        self.rows=rows
        self.cols=cols
        self.open_rows_per_run=open_rows_per_run
    
    def forward(self,x):
        pass

class NoiseCrossBar(CrossBar):
    def forward(self,x):
        with torch.no_grad():
            # thermal noise, shot noise
            BITWIDTH=2
            voltage_drop=0.2
            frequency=100*10e6 # 100M
            K_B=1.38e-23  # Boltzmann const
            temp=300 # temperature in kelvin
            delta_G=0.000333/2**(BITWIDTH-1)
            q = 1.6e-19  # electron charge
            G=w_q.abs()*delta_G
            sigma=torch.sqrt(G*frequency)*((4*K_B*temp+2*q*voltage_drop))**0.5/voltage_drop
            thermal_shot_noise=torch.randn_like(w_q)*sigma
            # random telegraph noise
            rtn_a = 1.662e-7  # RTN fitting parametera
            rtn_b = 0.0015  # RTN fitting parameter
            rtn_tmp=(rtn_b*G+rtn_a)
            G_rtn=G*rtn_tmp/(G-rtn_tmp)
            RTN_noise=G_rtn*torch.randint_like(G_rtn,0,2).float()
        return x