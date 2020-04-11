from sklearn.metrics import mean_squared_error,accuracy_score,auc,f1_score
import torch
import numpy as np

def rmse(y_true,y_pred):
    if isinstance(y_true,torch.Tensor):
        return ((y_pred-ytrue)**2).mean().pow(0.5).item()
    elif isinstance(y_true,np.ndarray):
        return mean_squared_error(y_true,y_pred,squared=False)
    
def accuracy(y_true,y_pred,reduction=True):
    if isinstance(y_true,torch.Tensor):
        y_pred = torch.argmax(y_true,dim=-1)
        if reduction:
            return y_pred.eq(y_true.view_as(preds)).mean().item()
        else:
            return y_pred.eq(y_true.view_as(preds)).sum().item()
        
    elif isinstance(y_true,np.ndarray):
        y_pred = np.argmax(y_pred,axis=-1)
        if reduction:
            return accuracy_score(y_true,y_pred)
        else:
            return accuracy_score(y_true,y_pred,normalize=False)

class f1(object):
    def __init__(self,average='macro'):
        self.average=average
    def __call__(self,y_true,y_pred):
        if isinstance(y_true,torch.Tensor):
            preds = y_pred.cpu().detach().numpy()
            truths = y_true.cpu().detach().numpy()
            
        elif isinstance(y_true,list):
            preds = np.array(y_preds)
            truths = np.array(y_true)
        else:
            preds = y_pred
            truths = y_true
        preds = np.argmax(preds,axis=1)
        return f1_score(truths,preds,average = self.average)
    @property
    def __name__(self):
        return self.__class__.__name__

_metrics = {'accuracy':accuracy,'f1_score':f1,'rmse':rmse}

def get_metric(name=''):
    
    if name not in _metrics:
        raise ValueError('Metric should be either accuracy,f1_score and rmse')
    return _metrics[name]                     