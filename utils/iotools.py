import csv
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torch
from .inference_transforms import Normalize, Pad

def read_path_csv(path):
    # read header, constract a dict 
    # each row correspond to a pathlist
    out_paths={}
    with open(path) as csvfile:
        path_reader = csv.DictReader(csvfile)
        header = path_reader.fieldnames
        for field in header:
            out_paths[field]=[]
        for row in path_reader:
            for key, val in row.items():
                out_paths[key].append(val)
    return out_paths

def read_inference_sample(path, stats=None):
    #read from disk
    img = Image.open(path)
    # apply transformations
    input_transforms = transforms.Compose([transforms.ToTensor(), Normalize(stats)])
    img_tensor = input_transforms(img)
    # output it as single batch tensor
    return img_tensor.unsqueeze(0), img.size[::-1]


def prepare_cv_frame(cv_rgb_img, norm_stats=None):
    #TODO add nomalization with ris and update figures
    input_transforms = transforms.Compose([myToTensor(),Normalize(norm_stats),Pad()])
    img_tensor = input_transforms(cv_rgb_img)
    return img_tensor.unsqueeze(0)


class myToTensor():
    def __call__(self, npimg):
        return torch.Tensor(npimg.astype(np.float32)/255.0).permute(2,0,1)
    


def save_disp(out_path, disparity, size=None, scale_factor=1):
    # i need to pass the original input size so that i can crop the output before saving it.
    # size =(h,w)
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    disparity[disparity<0]=0
    disp_np=disparity[0,0].detach().cpu().numpy()

    if size is not None:
        disp_np = disp_np[:size[0],:size[1]]

    #multipling by 256 and saving as a uint16 png we keep subpixel information
    out_img = Image.fromarray((disp_np*scale_factor).astype(np.uint16))
    out_img.save(out_path)

def save_seg(out_path, segmentation_logits, size=None, threshold=0.5):
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    seg_np=torch.sigmoid(segmentation_logits)[0,0].detach().cpu().numpy()
    seg_np[seg_np>threshold]=1
    seg_np[seg_np<=threshold]=0

    if size is not None:
        seg_np = seg_np[:size[0],:size[1]]

    out_img = Image.fromarray((seg_np*255).astype(np.uint8))
    out_img.save(out_path)

def post_process_disparity(disparity, scale_factor=1, size=None, bit_16=False):
    disparity[disparity<0]=0
    disp_np=disparity[0,0].detach().cpu().numpy()
    disp_np=disp_np*scale_factor

    if size is not None:
        disp_np = disp_np[:size[0],:size[1]]
    if bit_16:
        disp_np = disp_np.astype(np.uint16)
    else:
        disp_np = disp_np.astype(np.uint8)
    return disp_np

def post_process_segmentation(segmentation_logits, size=None, threshold=0.5):
    seg_np=torch.sigmoid(segmentation_logits)[0,0].detach().cpu().numpy()
    seg_np[seg_np>threshold]=1
    seg_np[seg_np<=threshold]=0

    if size is not None:
        seg_np = seg_np[:size[0],:size[1]]

    return (seg_np*255).astype(np.uint8)