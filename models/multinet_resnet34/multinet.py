from .feature_extractor_resnet34 import FeatureExtractor34
from .disparity_head import DispHead
import torch.nn as nn
from .unet_head import UNetHead

class Multinet(nn.Module):
    def __init__(self, max_disparity, n_classes, mode='multitask'):
        super().__init__()

        self.mode= mode
        assert self.mode in ['encoder', 'multitask', 'disparity', 'segmentation']


        self.feature_extractor = FeatureExtractor34()
        self.disparity_head = DispHead(max_disparity)
        self.segmentation_head = UNetHead(n_classes)


    def forward(self, img_left, img_right):

        disparity_scales = None
        segmentation_output = None

        left_features = self.feature_extractor(img_left)

        if self.mode =='encoder':
            return disparity_scales, segmentation_output

        if self.mode !='segmentation':
            right_features = self.feature_extractor(img_right)

            disparity_scales = self.disparity_head((img_left, *left_features),
                                               (img_right, *right_features))

        if self.mode !='disparity':
            segmentation_output = self.segmentation_head(left_features)

        return disparity_scales, segmentation_output