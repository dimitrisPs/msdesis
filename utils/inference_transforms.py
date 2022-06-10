import torch
import torch.nn.functional as F
from torchvision import transforms



class Pad(object):
    def __init__(self, pad_step=32):
        self.step=pad_step

    def __call__(self, img_tensor):
        # if needed, pad the tensor so that the width and high is divisible by 32
        h,w = img_tensor.size()[-2:]
        h_pad = self.step - (h%self.step) if h%self.step!=0 else 0
        w_pad = self.step - (w%self.step) if w%self.step!=0 else 0

        if h_pad==0 and w_pad==0 :
            return img_tensor
        else:
            # (padding_left,padding_right, padding_top, padding_bottom)
            return F.pad(img_tensor, (0,w_pad,0, h_pad))

class Normalize(object):
    def __init__(self, stats=None):
        self.stats=stats


    def __call__(self, img_tensor):
        # compute mean and std of sample and use them to nomalize 
        
        if self.stats is None:
            stats = torch.std_mean(img_tensor, dim=(1,2))
        else:
            stats= self.stats
        img_tensor = transforms.Normalize(stats[1], stats[0])(img_tensor)
        return img_tensor