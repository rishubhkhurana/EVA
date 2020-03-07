import torchvision.utils as vutils
import matplotlib.pyplot as plt



def show_batch(dl,rows=4,figsize=(4*3,4*3),padding=2,normalize=True):
    """
    plot a batch of images
    """
    plt.figure(figsize=figsize)
    data = next(iter(dl))
    x,_ = data[:rows**2]
    imgs = vutils.make_grid(x,padding=padding,normalize=normalize).numpy().transpose(1,2,0)
    plt.imshow(imgs)
    plt.axis('off')
    
    

