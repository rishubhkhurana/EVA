import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,bn=True,activation=True,pre_activation=False,kernel_size=(3,3),stride=(1,1),dilation=1,groups=1,drop=False,p=0.1,use_bias=False):
        super().__init__()
        lyrs =[]
        if pre_activation:
            if bn:
                lyrs.append(nn.BatchNorm2d(in_channels))
            lyrs.append(nn.ReLU())
            if drop:
                lyrs.append(nn.Dropout(p=p))
            
        lyrs.append(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,bias=~bn,padding=kernel_size[0]//2,
                         dilation=dilation,groups=groups))
        if not pre_activation:
            if bn:
                lyrs.append(nn.BatchNorm2d(out_channels))
            if activation:
                lyrs.append(nn.ReLU())
            if drop:
                lyrs.append(nn.Dropout(p=p))
                            
                
        self.lyrs=nn.Sequential(*lyrs)
    def forward(self,x):
        return self.lyrs(x)

class DepthConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_per_layer=1,first_kernel_size=(3,3),**kwargs):
        super().__init__()
        lyrs=[]
        lyrs.append(ConvLayer(in_channels,in_channels*kernel_per_layer,bn=False,activation=False
                              ,kernel_size=first_kernel_size,groups=in_channels))
        lyrs.append(ConvLayer(in_channels*kernel_per_layer,out_channels,**kwargs))
        self.lyrs = nn.Sequential(*lyrs)
    def forward(self,x):
        return self.lyrs(x)
    
    

def get_conv(in_channels,out_channels,conv_typ='regular',**kwargs):
    if conv_typ=='regular':
        return ConvLayer(in_channels,out_channels,**kwargs)
    elif conv_typ=='depthwise':
        return DepthConv(in_channels,out_channels,**kwargs)
    elif conv_typ=='transpose':
        return
    
def prepare_convblock(types=['regular']*2,channels=[16,32],ks=[3,3]):
    return [(t,c,k) for t,c,k in zip(types,channels,ks)]    

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,bottleneck=False,drop=False,p=0.1):
        super().__init__()
        if in_channels!=out_channels:
            stride = 2
            self.shortcut = ConvLayer(in_channels,out_channels,stride=(2,2),kernel_size=(1,1),drop=False,activation=False)
        else:
            self.shortcut = nn.Sequential()
            
        if not bottleneck:
            self.conv = nn.Sequential(ConvLayer(in_channels,out_channels,drop=drop,p=p,stride=stride)
                                      ,ConvLayer(out_channels,out_channels,drop=drop,p=p,activation=False))
        else:
            self.conv = nn.Sequential(ConvLayer(in_channels,in_channels//4,drop=drop,p=p,stride=stride),
                                      ConvLayer(in_channels//4,in_channels//4,drop=drop,p=p),
                                      ConvLayer(in_channels//4,out_channels,drop=drop,p=p,activation=False))
    def forward(self,x):
        y = self.conv(x)
        x = self.shortcut(x)
        
        return F.relu(x + y)


    

                         
            
            
    
                 