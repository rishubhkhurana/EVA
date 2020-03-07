import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from functools import partial


def get_rotate(angle=5):
    return transforms.RandomRotation((-angle,angle))
def get_normalize(mean=[0.5]*3,std=[0.5]*3):
    return transforms.Normalize(mean,std)
def get_tensor():
    return transforms.ToTensor()
def get_flip(typ='horizontal'):
    if typ=='horizontal':
        return transforms.RandomHorizontalFlip()
    else:
        return transforms.RandomVerticalFlip()

def create_transform(tfm_names,**kwargs):
    tfms = []
    if 'angle' in kwargs:
        angle = kwargs['angle']
    if 'norm_constants' in kwargs:
        u,s = kwargs['norm_constants']
    
    for t in tfm_names:
        if t=='rotate':
            tfms.append(_transforms[t](angle=angle) if 'angle' in kwargs else _transforms[t]())
        elif t=='normalize':
            tfms.append(_transforms[t](mean=u,std=s) if 'norm_constants' in kwargs else _transforms[t]())         
        else:
            tfms.append(_transforms[t]())
    #return tfms
    return transforms.Compose(tfms)
        
_transforms={'tensor':get_tensor,'rotate':get_rotate,'normalize':get_normalize,'hflip':partial(get_flip,'horizontal'),'vflip':partial(get_flip,'vertical')}

class VisionData:
    def __init__(self,name='CIFAR10'):
        self.name = name
    def load(self,tfms=[],train=True):
        if self.name=='CIFAR10':
            data = dset.CIFAR10(root='./data',transform=tfms,download=True,train=train)
        elif self.name=='MNIST':
            data = dset.MNIST(root='./data',transform=tfms,download=True,train=train)            
        return data

def getDataLoader(dataset,**kwargs):
    return DataLoader(dataset,**kwargs)


