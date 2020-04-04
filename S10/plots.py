import torchvision.utils as vutils
import matplotlib.pyplot as plt



def show_batch(dl,rows=4,figsize=(4*3,4*3),padding=2,normalize=True):
    """
    plot a batch of images
    """
    plt.figure(figsize=figsize)
    data = next(iter(dl))
    x=data[0][:rows**2]
    imgs = vutils.make_grid(x,padding=padding,normalize=normalize).numpy().transpose(1,2,0)
    plt.imshow(imgs)
    plt.axis('off')
    

def plot_diagnostics(err_dict):
    tlosses = err_dict['TrainingLosses']
    vlosses = err_dict['ValidationLosses']
    tacc = err_dict['TrainingAccuracy']
    vacc = err_dict['ValidationAcc']
    fig,(axs1,axs2) = plt.subplots(1,2,figsize=(16,8))
    axs1.plot(tlosses,label='Training Losses')
    axs1.plot(vlosses,label='Validation Losses')
    axs1.legend(loc='best')
    axs1.set_xlabel('Epoch')
    axs1.set_ylabel('Loss')
    axs2.plot(tacc,label='Training Accuracy')
    axs2.plot(vacc,label='Validation Accuracy')
    axs2.legend(loc='best')
    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


