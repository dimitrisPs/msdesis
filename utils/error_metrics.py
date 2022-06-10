import torch


def mIoU(prediction, target):
    # prediction (BS, CL, W,H)
    # target (BS, CL, W, H)
    # for now CL =1
    assert prediction.size() == target.size()
    assert len(prediction.size())==4
    
    eps = 1e-6
    
    
    intersection = ((prediction==1) & (target==1)).float().sum((2, 3))
    union = ((prediction==1) | (target==1)).float().sum((2, 3))

    iou = (intersection + eps) / (union + eps)
    
    return iou.mean()

def disparity_epe(prediction, reference, max_disparity=None):

    assert prediction.size() == reference.size()
    assert len(prediction.size())==4
    # find all valid gt values
    
    if max_disparity:
        valid_mask = (reference>0) & (reference < max_disparity)
    else:
        valid_mask = reference>0
        
    diff = torch.abs(prediction - reference)
    diff[~valid_mask] =0
    valid_pixels = torch.sum(valid_mask, axis=(1,2,3))
    err = torch.sum(diff, axis=(1,2,3))
    batch_error = err/valid_pixels
    #regect samples with no disparity values out of range.
    return torch.mean(batch_error[valid_pixels>0]).detach()

def disparity_bad3(prediction, reference, max_disparity=None):

    assert prediction.size() == reference.size()
    assert len(prediction.size())==4
    # find all valid gt values
    
    if max_disparity:
        valid_mask = (reference>0) & (reference < max_disparity)
    else:
        valid_mask = reference>0
        
    diff = torch.abs(prediction - reference)
    diff[~valid_mask]=0
    bad_px = diff>3
    valid_pixels = torch.sum(valid_mask, axis=(1,2,3))
    bad_px = torch.sum(bad_px, axis=(1,2,3))

    err = (bad_px/valid_pixels)*100
    return err.mean().detach()