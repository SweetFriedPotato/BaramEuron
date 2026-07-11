import torch
from torch import nn
class MLPNet(nn.Module):
    def __init__(self,input_size,hidden_dims=(128,64),dropout=.1):
        super().__init__(); layers=[]; n=input_size
        for h in hidden_dims: layers += [nn.Linear(n,h),nn.ReLU(),nn.Dropout(dropout)]; n=h
        layers += [nn.Linear(n,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

