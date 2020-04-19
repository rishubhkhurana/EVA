from layers import *
import torch.nn as nn
import torch.nn.functional as F
import torch




class ResNetCIFAR90(nn.Module):
    
    def __init__(self,in_channels=3,nc=10):
        super().__init__()
        self.nc=10
        self.prep_layer = nn.Sequential(*[nn.Conv2d(3,64,kernel_size=(3,3),stride=1,bias=False,padding=1),nn.BatchNorm2d(64),nn.ReLU()])
        self.cblock1_shortcut = nn.Sequential(*[nn.Conv2d(64,128,kernel_size=(3,3),stride=1,padding=1,bias=False),
                                                nn.MaxPool2d((2,2)),nn.BatchNorm2d(128),nn.ReLU()])
        self.cblock1_resblock = ResBlock(128,128)
        
        self.cblock2 = nn.Sequential(*[nn.Conv2d(128,256,kernel_size=(3,3),stride=1,bias=False,padding=1),
                                       nn.MaxPool2d((2,2)),nn.BatchNorm2d(256),nn.ReLU()])
        self.cblock3_shortcut = nn.Sequential(*[nn.Conv2d(256,512,kernel_size=(3,3),stride=1,padding=1,bias=False),
                                                nn.MaxPool2d((2,2)),nn.BatchNorm2d(512),nn.ReLU()])
        self.cblock3_resblock = ResBlock(512,512)
        self.GMP = nn.MaxPool2d((4,4))
        self.out_layer = nn.Conv2d(512,nc,bias=False,kernel_size=(1,1),stride=1)
        
    def forward(self,x):
        x = self.prep_layer(x)
        x = self.cblock1_shortcut(x)
        y = self.cblock1_resblock(x)
        x = x+y
        x = self.cblock2(x)
        x = self.cblock3_shortcut(x)
        y = self.cblock3_resblock(x)
        x=x+y
        x = self.GMP(x)
        x = self.out_layer(x)
        x = x.view(-1,self.nc)
        return x

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
    
class ResNetImageNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=512,nc=10,blocks=[2,2,2,2],bottleneck=False,drop=False,p=0.1):
        super().__init__()
        assert len(blocks)==4, "Please make sure that we have only four blocks Gradients->Textures->Patterns->Part of Objects"
        # estimate the start channels as final channel count / 8 
        out_channels = out_channels//8
        self.bottleneck,self.drop,self.p,self.nc = bottleneck,drop,p,nc
        lyrs=[]
        lyrs.append(ConvLayer(in_channels,out_channels,drop=drop,p=p))
        lyrs.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
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


class CNNDepthWiseDilated(nn.Module):
    def __init__(self,p=0.1):
        super().__init__()
        # First Conv Block -- edges and gradients
        cblock1 =[]
        cblock1.append(get_conv(3,32,drop=True,p=p,conv_typ='depthwise',kernel_size=(1,1)))# 32,32,3
        cblock1.append(get_conv(32,64,drop=True,p=p))# 32,32,5
        self.cblock1 = nn.Sequential(*cblock1)
        tblock1=[]
        tblock1.append(nn.MaxPool2d((2,2)))# 32,16,6
        tblock1.append(get_conv(64,16,kernel_size=(1,1),drop=True,p=p))# 16,16,6 
        self.tblock1 = nn.Sequential(*tblock1)
        # Second Conv block -- textures and patterns
        cblock2 =[]
        cblock2.append(get_conv(16,64,drop=True,p=p,dilation=2))# 16,16,14
        cblock2.append(get_conv(64,64,drop=True,p=p))# 16,16,18
        self.cblock2 = nn.Sequential(*cblock2)
        tblock2=[]
        tblock2.append(nn.MaxPool2d((2,2)))# 8,8,20
        tblock2.append(get_conv(64,16,kernel_size=(1,1),drop=True,p=p))# 8,8,20
        self.tblock2 = nn.Sequential(*tblock2)
        # Third Conv block -- part of objects 
        cblock3 =[]
        cblock3.append(get_conv(16,128,drop=True,p=p))# 8,8,28
        cblock3.append(get_conv(128,128,drop=True,p=p))# 8,8,36
        self.cblock3 = nn.Sequential(*cblock3)
        tblock3=[]
        tblock3.append(nn.MaxPool2d((2,2)))# 8,4,40
        tblock3.append(get_conv(128,32,kernel_size=(1,1),drop=True,p=p))# 4,4,40
        self.tblock3 = nn.Sequential(*tblock3)
        # Fourth Conv block -- objects
        cblock4 =[]
        cblock4.append(get_conv(32,256,drop=True,p=p))# 4,4,56
        #cblock4.append(get_conv(256,256))
        self.cblock4 = nn.Sequential(*cblock4)
        self.tblock4 = nn.AdaptiveAvgPool2d(1)
        #self.conv_blocks = nn.Sequential(self.cblock1,self.tblock1,self.cblock2,self.tblock2
        #                                 ,self.cblock3,self.tblock3,self.cblock4,self.tblock4)
        self.out_block = nn.Conv2d(256,10,kernel_size=(1,1))
        self.nc = 10
    def forward(self,x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.tblock3(x)
        x = self.cblock4(x)
        x = self.tblock4(x)
        x = self.out_block(x)
        
        return x.view(-1,self.nc)

class MNIST10K(nn.Module):
    def __init__(self,p=0.1,use_bias=True):
        super().__init__()
        self.alllyrs=nn.Sequential()
        cblock1 = []
        cblock1.append(nn.Sequential(*[nn.Conv2d(1,12,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=26,RF=3
        cblock1.append(nn.Sequential(*[nn.Conv2d(12,24,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(24),nn.ReLU(),nn.Dropout(p=p)])) # output_size=24,RF=5
        self.cblock1 = nn.Sequential(*cblock1)
        tblock1=[]
        tblock1.append(nn.MaxPool2d((2,2))) # output_size=12, RF=6
        tblock1.append(nn.Sequential(*[nn.Conv2d(24,12,kernel_size=(1,1),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=12,RF=6
        self.tblock1=nn.Sequential(*tblock1)
        cblock2 = []
        cblock2.append(nn.Sequential(*[nn.Conv2d(12,12,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=10,RF=10
        cblock2.append(nn.Sequential(*[nn.Conv2d(12,12,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=8,RF=14
        cblock2.append(nn.Sequential(*[nn.Conv2d(12,12,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=6,RF=18
        cblock2.append(nn.Sequential(*[nn.Conv2d(12,12,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=4,RF=22
        cblock2.append(nn.Sequential(*[nn.Conv2d(12,12,kernel_size=(3,3),stride=1,bias=use_bias),nn.BatchNorm2d(12),nn.ReLU(),nn.Dropout(p=p)])) # output_size=4,RF=26

        self.cblock2 = nn.Sequential(*cblock2)
        tblock2=[]
        tblock2.append(nn.AdaptiveAvgPool2d((1,1)))
        tblock2.append(nn.Conv2d(12,10,kernel_size=(1,1),stride=1))
        self.tblock2=nn.Sequential(*tblock2)
    def forward(self,x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        return F.log_softmax(x).squeeze().squeeze()
