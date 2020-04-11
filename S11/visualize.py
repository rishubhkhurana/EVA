import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
import torch

class GradCAM:
    
    def __init__(self,model,norm_constants=([0.5]*3,[0.5]*3),image_size=(32,32)):
        self.loc,self.scale = np.array(norm_constants[0])[None,None,:],np.array(norm_constants[1])[None,None,:]
        self.model = model
        self.model.eval()
        self.forwards={}
        self.backwards={}    
        self.fhooks={}
        self.bhooks={}
        self.register_all()
        self.image_size= image_size
    def forward(self,module,inp,output):
        self.forwards[id(module)]=output.data.cpu()
        
    def backward(self,module,grad_in,grad_out):
        self.backwards[id(module)]=grad_out[0].cpu()
     
    def find(self,layer_name):
        for k,v in self.forwards.items():
            for name,module in self.model.named_modules():
                if id(module)==k:
                    if name==layer_name:
                        fmap=v
                        grad=self.backwards[id(module)]
                        return fmap,grad
        print("layer name not founf")
        return "layer name not found"
        
        
    def register_all(self):
        for name,module in self.model.named_modules():
            self.fhooks[id(module)]=module.register_forward_hook(self.forward)
            self.bhooks[id(module)]=module.register_backward_hook(self.backward)
            
    def __call__(self,img,layer_name,class_num,show=False,ax=None):
        # estimate model output
        preds=self.model(img)
        # convert the output of model into probabilities
        probs = F.softmax(preds,dim=1)
        # get the one hot ideal output
        one_hot = np.zeros((1, probs.size()[-1]), dtype=np.float32)
        one_hot[0][class_num] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * probs)
        # zero out the gradients and then call the backward
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # get the forward activation and gradient of the layer 
        fmap,grad=self.find(layer_name)
        grads_val = grad.data.numpy()
        
        # get the target
        target = fmap.data.numpy()[0,:]
        # calculate the gap on every gradient maps
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        # create a placeholder for class activation map
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        
        # multiply each forward activation maps with their respective grad weights and then sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # apply ReLU to class activation so as to only keep positive correlation
        cam = np.maximum(cam, 0)
        # resize the cam to the image size
        cam = cv2.resize(cam, self.image_size)
        # normalize the sam
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # if we are required to show then de normalize the actual image to show
        if show:
            show_img = img.numpy().squeeze(0).transpose((1,2,0))
            show_img = show_img*self.scale+self.loc
            ax=self.show_cam_on_image(show_img,cam,ax=ax)
        return ax
        
    
    def remove_hooks(self):
        for k,v in self.fhooks:
            v.remove()
        for k,v in self.bhooks:
            v.remove()
        
    def show_cam_on_image(self,img, mask,ax=None):
        fig =None
        if ax is None:
            fig,ax = plt.subplots()
        # create heatmap from cam and rescale it to 255 
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        # normalize the heatmap back
        heatmap = np.float32(heatmap) / 255
        # compose the cam over image
        cam = heatmap + np.float32(img)
        # normalize and show
        cam = cam / np.max(cam)
        cam = np.uint8(255*cam)
        ax.imshow(cam)
        ax.axis('off')
        if fig is not None:
            plt.show()
        return ax
        
        
        
        