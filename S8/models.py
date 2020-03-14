from layers import *
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=512,nc=10,blocks=[2,2,2,2],bottleneck=False,drop=False,p=0.1):
        super().__init__()
        assert len(blocks)==4, "Please make sure that we have only four blocks Gradients->Textures->Patterns->Part of Objects"
        # estimate the start channels as final channel count / 8 
        out_channels = out_channels//8
        self.bottleneck,self.drop,self.p,self.nc = bottleneck,drop,p,nc
        lyrs=[]
        lyrs.append(ConvLayer(in_channels,out_channels,drop=drop,p=p))
        in_channels=out_channels
        # design block by block
        for b in blocks:
            lyrs.append(self._make_block(in_channels,out_channels,b))
            in_channels = out_channels
            out_channels*=2
        out_channels=out_channels//2
        lyrs.append(nn.AdaptiveAvgPool2d((1,1)))
        lyrs.append(ConvLayer(out_channels,nc,drop=False,activation=False,bn=False,kernel_size=(1,1)))
        self.net = nn.Sequential(*lyrs) 
        
    def _make_block(self,in_channels,out_channels,blocks):
        lyrs=[]
        for i in range(blocks):
            if i==0:
                lyrs.append(ResBlock(in_channels,out_channels,bottleneck=self.bottleneck,drop=self.drop,p=self.p))
                continue
            lyrs.append(ResBlock(out_channels,out_channels,bottleneck=self.bottleneck,drop=self.drop,p=self.p))
        return nn.Sequential(*lyrs)
    
    def forward(self,x):
        x = self.net(x)
        x = x.view(-1,self.nc)
        return x
    
    
