import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from functools import partial
import torch
from albumentations import HorizontalFlip, HueSaturationValue, Rotate,RandomSizedCrop, Compose, Resize, PadIfNeeded,Normalize,VerticalFlip,Cutout
from albumentations.pytorch import ToTensor
import cv2

def get_rotate(angle=5):
    return Rotate(limit=angle,border_mode=2)
def get_normalize(mean=[0.5]*3,std=[0.5]*3):
    return Normalize(mean,std)
def get_padding(size=(40,40)):
    return PadIfNeeded(min_height=size[0],min_width=size[1])
def get_tensor():
    return ToTensor()
def get_crop(size=(32,32),crop_padding=4):
    return RandomSizedCrop(size,size[0],size[1])
def get_flip(typ='horizontal'):
    if typ=='horizontal':
        return HorizontalFlip()
    else:
        return VerticalFlip()  
def get_cutout(size=16,fill_value=[0.5]*3,p=0.5):
    return Cutout(num_holes=1,max_h_size=size,max_w_size=size,fill_value=fill_value,p=p)
def get_jitter():
    return HueSaturationValue()

def create_transform(tfm_names,**kwargs):
    
    tfms = []

    for t in tfm_names:
        params={}
        if t=='rotate':
            if 'angle' in kwargs:
                params['angle']=kwargs['angle']
            tfms.append(_transforms[t](**params))
        elif t=='normalize':
            if 'norm_constants' in kwargs:
                params['mean'],params['std'] = kwargs['norm_constants']
            tfms.append(_transforms[t](**params)) 
        elif t=='crop':
            if 'crop_padding' in kwargs:
                params['crop_padding'] = kwargs['crop_padding']
            if 'crop_size' in kwargs:
                params['size'] = kwargs['crop_size']
            tfms.append(_transforms[t](**params))
        elif t=='pad':
            if 'padded_size' in kwargs:
                params['size'] = kwargs['padded_size']
            tfms.append(_transforms[t](**params))
        elif t=='cutout':
            if 'cutout_size' in kwargs:
                params['size'] = kwargs['cutout_size']
            if 'cutout_fillvalue' in kwargs:
                params['fill_value'] = kwargs['cutout_fillvalue']
            if 'cutout_prob' in kwargs:
                params['p']=kwargs['cutout_prob']
            tfms.append(_transforms[t](**params))
        else:
            tfms.append(_transforms[t]())

    return Compose(tfms)
        
_transforms={'tensor':get_tensor,'rotate':get_rotate,'normalize':get_normalize,'hflip':partial(get_flip,'horizontal'),'vflip':partial(get_flip,'vertical'),'crop':get_crop,'cutout':get_cutout,'jitter':get_jitter,'pad':get_padding}

class VisionData:
    def __init__(self,images,labels,tfms=[]):
        self.images = images
        self.labels=labels
        self.tfms=tfms
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        # reading data
        img = self.images[idx]
        target = self.labels[idx]
        # augmenting data
        if self.tfms:
            img = self.tfms(image=img)['image']
        return img,target
    @classmethod
    def load(cls,name='CIFAR10',tfms=[],train=True):
        if name=='CIFAR10':
            data = dset.CIFAR10(root='./data',download=True,train=train)
        elif name=='MNIST':
            data = dset.MNIST(root='./data',download=True,train=train)            
        return VisionData(data.data,data.targets,tfms=tfms)

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
