import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # due to issues with interpolating non dense reference disparities,
        # we upscale the indermidiate inputs and compute loss in original resolution
        for i in range(self.supervision_levels):

            # if self.weights[i] ==0:
            #     continue
            output = prediction[i]
            scale = 2**i
            if scale!=1:
                output = F.interpolate(output, scale_factor=scale, align_corners=True, mode='bilinear')*scale
            
            self.output_scaled.append(output)
            loss += F.smooth_l1_loss(output[mask], reference[mask], reduction='mean') * self.weights[i]

        return loss