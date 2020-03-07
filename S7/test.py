import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook

def test(mod,dl,loss_func,device='cpu'):
    totloss=0
    totacc=0
    bcount=0
    mod.eval()
    with torch.no_grad():
        for xb,yb in tqdm_notebook(dl):
            xb,yb = xb.to(device),yb.to(device)
            bs = xb.shape[0]
            preds = mod(xb)
            loss = loss_func(preds,yb)
            totloss+=loss.item()*bs
            bcount+=bs
            y_preds = preds.argmax(dim=1,keepdim=True)
            acc = y_preds.eq(yb.view_as(y_preds)).sum().item()
            totacc+=acc
    totloss/=bcount
    totacc/bcount
    return {'Loss':totloss,'Accuracy':100*totacc/bcount}


def top5errors(mod,dl):
    pass