import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,tqdm_notebook
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,_LRScheduler,ReduceLROnPlateau,OneCycleLR,CyclicLR
import matplotlib.pyplot as plt
from metrics import *

_schedulers = {'step':StepLR,'reduce':ReduceLROnPlateau,'onecycle':OneCycleLR,'cyclic':CyclicLR}

def train(mod,opt,dls,loss_func,n_epochs,scheduler=None,device='cuda',sched_loss=False,sched_batchwise=False):
    tlosses_batchwise=[]
    vlosses_batchwise=[]
    tlosses=[]
    vlosses=[]
    tacc=[]
    vacc=[]
    
    for e in range(n_epochs):
        # walk through one epoch of training
        ldict=one_epoch(mod,opt,dls['train'],loss_func,train=True,device=device,
                       scheduler=None if not sched_batchwise else scheduler)
        tlosses_batchwise.extend(ldict['Losses'])
        tlosses.append(ldict['AvgLoss'])
        tacc.append(ldict['Accuracy'])
        # walk through one epoch of validation
        ldict=one_epoch(mod,opt,dls['valid'],loss_func,train=False,device=device)   
        if (scheduler is not None) and (not sched_batchwise):
            if not sched_loss:
                scheduler.step()
            else:
                scheduler.step(ldict['AvgLoss'])
        vlosses_batchwise.extend(ldict['Losses'])
        vlosses.append(ldict['AvgLoss'])
        vacc.append(ldict['Accuracy'])
        print(f'[{e}/{n_epochs}]--> Training Loss:{tlosses[-1]:.3f}, Training Accuracy:{100*tacc[-1]:.3f}, Validation Loss:{vlosses[-1]:.3f}, Validation Accuracy:{100*vacc[-1]:.3f}')
        print(f'Learning rate: {opt.param_groups[-1]["lr"]}')
    return {'TrainingLosses':tlosses,'ValidationLosses':vlosses,'TrainingAccuracy':tacc,'ValidationAcc':vacc,'TrainingBatchLosses':tlosses_batchwise,'ValidationBatchLosses':vlosses_batchwise}
        
def one_epoch(mod,opt,dl,loss_func,train=True,device='cuda',scheduler=None):
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
            if scheduler is not None:
                scheduler.step()
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
        
class ExponentialLR(_LRScheduler):
    def __init__(self,opt,end_lr,max_iters):
        self.end_lr = end_lr
        self.max_iters = max_iters
        super(ExponentialLR,self).__init__(opt,last_epoch=-1)
   
    def get_lr(self):
        curr_iter = self.last_epoch+1
        pos = curr_iter/self.max_iters
        return [base_lr*(self.end_lr/base_lr)**pos for base_lr in self.base_lrs]
    
class LinearLR(_LRScheduler):
    def __init__(self,opt,end_lr,max_iters):
        self.end_lr = end_lr
        self.max_iters = max_iters
        super(LinearLR,self).__init__(opt,last_epoch=-1)
   
    def get_lr(self):
        curr_iter = self.last_epoch+1
        pos = curr_iter/self.max_iters
        return [base_lr+(self.end_lr-base_lr)*pos for base_lr in self.base_lrs]       
    

class LRFinder(object):
    
    def __init__(self,model,opt,criterion,device=None):
        
        # register the optimizer and check if there is a scheduler attached to optimizer
        self.opt = opt
        self._check_for_scheduler()
        # register the model
        self.model = model
        self.criterion = criterion
        
        # save original state of model and optimizer
        self.model_device = next(self.model.parameters()).device # just pull one parameter and check its device
        
        # if device is None, the use model_device
        if device:
            self.device = device
        else:
            self.device = self.model_device
    
        
    def _check_for_scheduler(self):
        for param_group in self.opt.param_groups:
            if 'initial_lr' in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")
    
    def _set_learning_rate(self,lrs):
        if not isinstance(lrs,list):
            lrs = [lrs]*len(self.opt.param_groups)
        if len(lrs)!=len(self.opt.param_groups):
            raise ValueError("Length of lrs don't match length of param groups")
        
        for lr,pg in zip(lrs,self.opt.param_groups):
            pg['lr'] = lr
             
        
    
    def range_test(self,train_dl,valid_dl=None,start_lr=None,end_lr =10,num_iter = 100,
                   step_mode ='exp',smooth_frac = 0.05, diverge_th = 5,
                   accumulation_steps=1,metric=None,break_on_loss=True):
        
        # reset the history
        if metric is not None:
            self.history ={'lr':[],'loss':[],metric:[]}
            metric_func = get_metric(name=metric)
        else:
            self.history ={'lr':[],'loss':[]}
            metric_func=None
        self.best_loss = None
        
        # move the model to device
        self.model.to(self.device)
        
        # check for pre existing scheduler
        self._check_for_scheduler()
        
        # set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)
        
        # set the requested finder
        if step_mode.lower() == 'exp':
            stepper = ExponentialLR(self.opt,end_lr,num_iter)
        elif step_mode.lower() == 'linear':
            stepper = LinearLR(self.opt,end_lr,num_iter)
        
        else:
            raise ValueError("Expected one of (linear,exp) as step_mode")
        
        # check if smooth ratio is within 0 and 1 
        if smooth_frac<0 and smooth_frac>1:
            raise ValueError("smooth_frac should be between 0 and 1")
        dl = DataLoaderWrapper(train_dl) 
        for it in tqdm(range(num_iter)):
            loss,batch_metric=self.one_batch(dl,metric_func=metric_func)
            if valid_dl is not None:
                loss = self.validate(dl)
            stepper.step()
            self.history['lr'].append(stepper.get_lr()[0])
            
            # track the best loss
            if it ==0:
                self.best_loss = loss
            else:
                if smooth_frac>0:
                    loss = loss*smooth_frac + (1.-smooth_frac)*self.history['loss'][-1]
                if loss<self.best_loss:
                    self.best_loss=loss
            
            self.history['loss'].append(loss)
            if metric is not None:
                if it>0:
                    batch_metric = batch_metric*smooth_frac + (1-smooth_frac)*self.history[metric][-1]
                    
                self.history[metric].append(batch_metric)
            if (loss > self.best_loss*diverge_th) and break_on_loss:
                print("Stopping early as the loss has started to diverge")
                break
        print("Learning rate search is over. Please use self.plot to look at the graph") 
                
    def validate(self,dl,metric_func=None):
        self.model.eval()
        total_loss = 0
        batch_count=0
        with torch.no_grad():
            all_preds =[]
            all_gts=[]
            for xb,yb in dl:
                bs=xb.size(0)
                xb,yb = xb.to(self.device),yb.to(self.device)
                preds = self.model(xb)
                loss = self.criterion(preds,yb)
                all_preds.append(preds.cpu().numpy())
                all_gts.append(yb.cpu().numpy())
                total_loss+=loss.item()*bs
                batch_count+=bs
            all_preds = np.concatenate(all_preds)
            all_gts = np.concatenate(all_gts)
        return total_loss/batch_count,metric_func(all_gts,all_preds)
    
    def one_batch(self,dl,metric_func=None):
        self.model.train()
        xb,yb = dl.get_batch()
        xb,yb = xb.to(self.device),yb.to(self.device)
        preds = self.model(xb)
        loss = self.criterion(preds,yb)
        if metric_func is not None:
            metric = metric_func(yb.detach().cpu().numpy(),preds.detach().cpu().numpy())
        else:
            metric=None
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item(),metric
        
    def plot(self,skip_start=10,skip_end=10,ax=None,lr_scale='log',to_plot='loss'):
        if skip_start<0:
            raise ValueError("skip_start should be greater than 0")
        if skip_end<0:
            raise ValueError("skip_start should be greater than 0")
        lrs = self.history["lr"]
        
        if to_plot=='loss':
            key ='loss'
        else:
            key= [k for k in self.history.keys() if k not in ['loss','lr']][0]
        plot_y = self.history[key]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            plot_y = plot_y[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            plot_y = plot_y[skip_start:-skip_end]
        fig=None
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(lrs,plot_y)
        if lr_scale=='log':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(f'{key.upper()}')
        
        if fig is not None:
            plt.show()
        return ax
        
class DataLoaderWrapper:
    
    def __init__(self,dl):
        self.dl = dl
        self._iterator = iter(self.dl)
    def __next__(self):
        
        try:
            xb,yb = next(self._iterator)
        
        except StopIteration:
            self._iterator = iter(self.dl)
            xb,yb = next(self._iterator)
        finally:
            return xb,yb
    def get_batch(self):
        return self.__next__()
   
    
    
def get_scheduler(opt,typ,**kwargs):
    if typ not in _schedulers:
        raise ValueError(f"Please provide scheduler type in one of the {_schedulers.keys()}")
    return _schedulers[typ](opt,**kwargs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    