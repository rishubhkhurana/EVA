import numpy as np
from sklearn.metrics import confusion_matrix
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def show_matrix(model,dl):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for xb,yb in tqdm(dl):
            xb,yb = xb.to(device),yb.to(device)
            probs = model(xb)
            preds = probs.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_truths.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_truths = np.concatenate(all_truths)
    cm = confusion_matrix(all_preds,all_truths)
    plt.figure(figsize=(16,16))
    sns.heatmap(cm,annot=True)
    plt.axis('off')
    plt.show()
    return cm



def get_random_incorrects(model,dl,n_incorr=25):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for xb,yb in tqdm(dl):
            xb,yb = xb.to(device),yb.to(device)
            probs = model(xb)
            preds = probs.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_truths.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_truths = np.concatenate(all_truths)
    incorrect_idxs = np.where(all_preds!=all_truths)[0]
    return incorrect_idxs









