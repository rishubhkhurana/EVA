from layers import *


class QDNN(nn.Module):
    def __init__(self,in_channels=3,drop=False,p=0.1):
        super().__init__()
        from datetime import datetime
        print("Current Date/Time: ", datetime.now())
        self.conv1 = ConvLayer(in_channels,32,drop=drop,p=p)
        self.conv2 = ConvLayer(35,64,drop=drop,p=p)
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.conv3 = ConvLayer(99,16,kernel_size=(1,1),drop=drop,p=p)
        self.conv4 = ConvLayer(115,64,drop=drop,p=p)
        self.conv5 = ConvLayer(179,128,drop=drop,p=p)
        self.max_pool2 = nn.MaxPool2d((2,2))
        self.conv6 = ConvLayer(208,32,kernel_size=(1,1),drop=drop,p=p)
        self.conv7 = ConvLayer(240,64,drop=drop,p=p)
        self.conv8 = ConvLayer(304,128,drop=drop,p=p)       
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128,10)
    def forward(self,x1):
        x2 = self.conv1(x1)#32
        x3 = self.conv2(torch.cat([x1,x2],dim=1))#35
        x4 = self.max_pool1(torch.cat([x1,x2,x3],dim=1))#99
        x5 = self.conv3(x4)#16
        x6 = self.conv4(torch.cat([x4,x5],dim=1))#64
        x7 = self.conv5(torch.cat([x4,x5,x6],dim=1))#128
        x8 = self.max_pool2(torch.cat([x5,x6,x7],dim=1))#128+64+16=208
        x9 = self.conv6(x8)#32
        x10 = self.conv7(torch.cat([x8,x9],dim=1))#64
        x11 = self.conv8(torch.cat([x8,x9,x10],dim=1))#128
        x12 = self.avgpool(x11)#128
        x12 = x12.squeeze(-1).squeeze(-1)
        x13 = self.classifier(x12)
        return x13
    
        
        
                        
        