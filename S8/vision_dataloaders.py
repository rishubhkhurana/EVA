import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from functools import partial
import torch

def get_rotate(angle=5):
    return transforms.RandomRotation((-angle,angle))
def get_normalize(mean=[0.5]*3,std=[0.5]*3):
    return transforms.Normalize(mean,std)
def get_tensor():
    return transforms.ToTensor()
def get_crop(size=32,padding=4):
    return transforms.RandomCrop(size=size,padding=padding,padding_mode='reflect')
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
    if 'padding' in kwargs:
        padding = kwargs['padding']
    if 'size' in kwargs:
        size = kwargs['size']
        
    for t in tfm_names:
        if t=='rotate':
            tfms.append(_transforms[t](angle=angle) if 'angle' in kwargs else _transforms[t]())
        elif t=='normalize':
            tfms.append(_transforms[t](mean=u,std=s) if 'norm_constants' in kwargs else _transforms[t]())  
        elif t=='crop':
            tfms.append(_transforms[t](size=size,padding=padding) if 'padding' in kwargs else _transforms[t]())
        else:
            tfms.append(_transforms[t]())
    #return tfms
    return transforms.Compose(tfms)
        
_transforms={'tensor':get_tensor,'rotate':get_rotate,'normalize':get_normalize,'hflip':partial(get_flip,'horizontal'),'vflip':partial(get_flip,'vertical'),'crop':get_crop}

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

def getTrainTestLoader(datasets,BS=64):
    isCUDA = torch.cuda.is_available()
    dev = torch.device('cuda') if isCUDA else torch.device('cpu')
    dataloader_args = dict(shuffle=True,num_workers=4,batch_size = BS,pin_memory=True) if isCUDA else dict(shuffle=True,batch_size = BS//2)
    trn_dl = getDataLoader(datasets[0],**dataloader_args)
    dataloader_args = dict(shuffle=False,num_workers=4,batch_size = 2*BS,pin_memory=True) if isCUDA else dict(shuffle=False,batch_size = BS)
    tst_dl = getDataLoader(datasets[1],**dataloader_args)
    return trn_dl,tst_dl
