import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from .misc_layers import denormalize, disparity_warp

class l1_multiscale_loss(nn.Module):
    def __init__(self, supervision_levels, max_disparity, scale_weights=None, normalize_weights=True):
        super().__init__()
        self.max_disp = max_disparity
        self.weights = scale_weights
        self.supervision_levels=supervision_levels
        if (scale_weights is not None) and normalize_weights:
            self.weights = [w/sum(scale_weights) for w in scale_weights]


    def forward(self, prediction, reference, mask=None):
        assert prediction[0].size() == reference.size()
        assert len(reference.size())==4

        if self.weights is None:
            self.weights =[1/self.supervision_levels] * self.supervision_levels
        
        
        if mask ==None:
            mask = prediction[0].new_full((reference[0].shape), True, dtype=bool)

        mask = mask & (reference > 0) & (reference < self.max_disp)
        mask.detach_()
        loss = 0
        self.output_scaled=[]

        for i in range(self.supervision_levels):


            output = prediction[i]
            scale = 2**i
            if scale!=1:
                output = F.interpolate(output, scale_factor=scale, align_corners=False, mode='bilinear')*scale
            
            self.output_scaled.append(output)
            loss += F.smooth_l1_loss(output[mask], reference[mask], reduction='mean') * self.weights[i]

        return loss




class unsupervised_disparity_loss_multiscale(nn.Module):
    def __init__(self, photometric_weight=1, photometric_alpha=0.85, smoothness_weight=0.5, stats=None, ssim_window=11, multiscale_weights=[1]):
        super().__init__()
        self.w_smooth = smoothness_weight
        self.w_ph = photometric_weight

        self.stats=stats
        self.photometric_alpha= photometric_alpha
        self.crit_struct = kornia.losses.SSIMLoss(ssim_window)
        self.crit_smooth = kornia.losses.InverseDepthSmoothnessLoss()

        self.scale_weights=multiscale_weights

    def forward(self, disparity_scales:tuple, left:torch.Tensor, right:torch.Tensor):
        # denormalize the image tensors 
        assert isinstance(disparity_scales, tuple)
        assert len(self.scale_weights) == len(disparity_scales)

        if self.stats is not None:
            left = denormalize(left, self.stats)
            right = denormalize(right, self.stats)


        loss = 0 

        # upscale all disparity outputs

        for i,  disparity in enumerate(disparity_scales):
            
            scale = 2**i
            if self.scale_weights[i]==0:
                continue
            
            if i>0:
                disparity = F.interpolate(disparity, scale_factor=scale, mode='bilinear') * scale

            reconstructed_left = disparity_warp(right, disparity)
            

            photometric_loss = 0
            smoothness_loss = 0 
            if self.w_ph!=0:
                loss_ph=0
                loss_ssim=0
                if self.photometric_alpha!=1:
                    loss_ph = nn.SmoothL1Loss()(reconstructed_left, left)
                if self.photometric_alpha!=0:
                    loss_ssim = self.crit_struct(reconstructed_left, left)

                photometric_loss = self.photometric_alpha * loss_ssim + (1-self.photometric_alpha) * loss_ph

            if self.w_smooth!=0:
                normalized_disparity = disparity/disparity.mean(dim=(2,3), keepdims=True)
                smoothness_loss = self.crit_smooth(normalized_disparity, left)/scale

            loss += (self.w_ph * photometric_loss + self.w_smooth * smoothness_loss) * self.scale_weights[i]

        return loss