import torch
from torchvision import transforms
import numpy as np
import cv2



class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    
    def __call__(self, sample):
        
        left_img, right_img, gt = sample
        
        h, w = gt.shape[:2]
        new_h, new_w = self.output_size
        if (new_h==-1) and (new_w==-1):
            return sample
        if h == new_h:
            top=0
        else:
            top = np.random.randint(0, h - new_h)
        if w == new_w:
            left =0
        else:
            left = np.random.randint(0, w - new_w)
        
        left_img = left_img.crop((left, top, left+new_w, top+new_h))
        right_img = right_img.crop((left, top, left+new_w, top+new_h))
        gt = gt[top: top + new_h, left: left + new_w]
        
        return left_img, right_img, gt

    
class RandomFlipVertical(object):
    def __init__(self, p=0.5):
        assert((p>=0) and (p<=1))
        self.p=p
    def __call__(self, sample):
        
        left_img, right_img, gt = sample
        
        flip = torch.rand(1).item() < self.p
        
        if flip:
            left_img = transforms.functional.vflip(left_img)
            right_img = transforms.functional.vflip(right_img)
            gt = gt[::-1].copy()
            
        return left_img, right_img, gt
    

        
    
class ToTensor(object):
    def __call__(self, sample):
        left_img, right_img, gt = sample
        
        left_tensor_img = transforms.ToTensor()(left_img)
        right_tensor_img = transforms.ToTensor()(right_img)
        
        return left_tensor_img, right_tensor_img, gt

    
class Normalize(object):
    def __init__(self, statistics):
        self.normalize = transforms.Normalize(mean=statistics['mean'],std=statistics['std'])
    def __call__(self, sample):
        left_img_tensor, right_img_tensor, gt = sample
        
        left_img_tensor_norm = self.normalize(left_img_tensor)
        right_img_tensor_norm = self.normalize(right_img_tensor)
        
        
        return left_img_tensor_norm, right_img_tensor_norm, gt


class rescale(object):
    def __init__(self, factor):
        self.scale_factor = factor
    def __call__(self, sample):
        left_img, right_img, gt = sample
        if self.scale_factor==1:
            return sample

        h, w = gt.shape[:2]

        resize_t=transforms.Resize((h//self.scale_factor, w//self.scale_factor))

        left_img = resize_t(left_img)
        right_img = resize_t(right_img)

        gt = np.expand_dims(cv2.resize(gt, (w//self.scale_factor, h//self.scale_factor), interpolation = cv2.INTER_NEAREST),-1)
        # print(gt.shape)
        return left_img, right_img, gt

class NormalizeInverse(transforms.Normalize):
    """
    undo the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())