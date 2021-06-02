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