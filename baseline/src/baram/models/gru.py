import torch
from torch import nn
class GRUNet(nn.Module):
    def __init__(self,input_size,hidden_size=64,num_layers=1,dropout=0,bidirectional=False):
        super().__init__(); self.gru=nn.GRU(input_size,hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout if num_layers>1 else 0,bidirectional=bidirectional); self.head=nn.Linear(hidden_size*(2 if bidirectional else 1),1)
    def forward(self,x):
        out,_=self.gru(x); return self.head(out[:,-1]).squeeze(-1)

