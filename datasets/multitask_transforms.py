import torch
from torchvision import transforms
import numpy as np



class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    
    def __call__(self, sample):
        
        left_img, right_img, disp_gt, seg_gt = sample
        
        h, w = disp_gt.shape[:2]
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

        disp_gt = disp_gt[top: top + new_h, left: left + new_w]
        seg_gt = seg_gt[top: top + new_h, left: left + new_w]
        
        return left_img, right_img, disp_gt, seg_gt

    
class RandomFlipVertical(object):
    def __init__(self, p=0.5):
        assert((p>=0) and (p<=1))
        self.p=p
    def __call__(self, sample):
        
        left_img, right_img, disp_gt, seg_gt = sample
        
        flip = torch.rand(1).item() < self.p
        
        if flip:
            left_img = transforms.functional.vflip(left_img)
            right_img = transforms.functional.vflip(right_img)
            disp_gt = disp_gt[::-1].copy()
            seg_gt = seg_gt[::-1].copy()
            
        return left_img, right_img, disp_gt, seg_gt
    

class ToTensor(object):
    def __call__(self, sample):
        left_img, right_img, *gt = sample
        
        left_tensor_img = transforms.ToTensor()(left_img)
        right_tensor_img = transforms.ToTensor()(right_img)
        
        return left_tensor_img, right_tensor_img, *gt

    
class Normalize(object):
    def __init__(self, statistics):
        self.normalize = transforms.Normalize(mean=statistics['mean'],std=statistics['std'])
    def __call__(self, sample):
        left_img_tensor, right_img_tensor, *gt = sample
        
        left_img_tensor_norm = self.normalize(left_img_tensor)
        right_img_tensor_norm = self.normalize(right_img_tensor)
        
        
        return left_img_tensor_norm, right_img_tensor_norm, *gt