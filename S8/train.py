import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,tqdm_notebook

def train(mod,opt,dls,loss_func,n_epochs,scheduler=None,device='cuda'):
    tlosses_batchwise=[]
    vlosses_batchwise=[]
    tlosses=[]
    vlosses=[]
    tacc=[]
    vacc=[]
    for e in range(n_epochs):
        # walk through one epoch of training
        ldict=one_epoch(mod,opt,dls['train'],loss_func,train=True,device=device)
        if scheduler is not None:
            scheduler.step()
        tlosses_batchwise.extend(ldict['Losses'])
        tlosses.append(ldict['AvgLoss'])
        tacc.append(ldict['Accuracy'])
        # walk through one epoch of validation
        ldict=one_epoch(mod,opt,dls['valid'],loss_func,train=False,device=device)
        vlosses_batchwise.extend(ldict['Losses'])
        vlosses.append(ldict['AvgLoss'])
        vacc.append(ldict['Accuracy'])
        print(f'[{e}/{n_epochs}]--> Training Loss:{tlosses[-1]:.3f}, Training Accuracy:{100*tacc[-1]:.3f}, Validation Loss:{vlosses[-1]:.3f}, Validation Accuracy:{100*vacc[-1]:.3f}')
    return {'TrainingLosses':tlosses,'ValidationLosses':vlosses,'TrainingAccuracy':tacc,'ValidationAcc':vacc,'TrainingBatchLosses':tlosses_batchwise,'ValidationBatchLosses':vlosses_batchwise}
        
def one_epoch(mod,opt,dl,loss_func,train=True,device='cuda'):
    losses=[]
    bcount=0
    totloss=0
    totacc=0
    if train:
        pbar = tqdm(dl)
        mod.train()
        for i,data in enumerate(pbar):
            # get data
            xb,yb = data
            bs = xb.shape[0]
            # move data to the device 
            xb,yb = xb.to(device),yb.to(device)
            # zero out all grads
            opt.zero_grad()
            # predict on x
            preds = mod(xb)
            # estimate loss
            loss = loss_func(preds,yb)
            # call backward
            loss.backward()
            # take one optimizer step
            opt.step()
            # update total loss and batch count
            totloss+= loss.item()*bs
            losses.append(loss.item())
            bcount+=bs
            # estimate final predictions and accuracies
            y_pred = preds.argmax(dim=1,keepdim=True)
            acc=y_pred.eq(yb.view_as(y_pred)).sum().item()
            totacc+=acc
            pbar.set_description(f'Training Loss:{loss:.3f}, Training Acc:{100*acc/bs:.3f}')
        
    else:
        mod.eval()
        with torch.no_grad():
            for data in tqdm_notebook(dl):
                xb,yb = data
                bs=xb.shape[0]
                xb,yb = xb.to(device),yb.to(device)
                preds = mod(xb)
                loss = loss_func(preds,yb)
                totloss+=loss.item()*bs
                bcount+=bs
                y_preds = preds.argmax(dim=1,keepdim=True)
                acc = y_preds.eq(yb.view_as(y_preds)).sum().item()
                totacc+=acc
                losses.append(loss.item())
    totloss/=bcount
    totacc/=bcount
    return {'Losses':losses,'AvgLoss':totloss,'Accuracy':totacc}
        
    
    
    
    
    
    