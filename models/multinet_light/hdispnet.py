from .feature_extractor_madnet import FeaturePyramid
from .disparity_head import DispHead
import torch.nn as nn

class HDispNet(nn.Module):
    def __init__(self, max_disparity, pretrained_features=False):
        super().__init__()
        print(max_disparity, pretrained_features)
        self.feature_extractor = FeaturePyramid()
        self.disparity_head = DispHead(max_disparity)


    def forward(self, img_left, img_right):


        left_features = self.feature_extractor(img_left)
        right_features = self.feature_extractor(img_right)

        disparity_scales = self.disparity_head((img_left, *left_features),
                                               (img_right, *right_features))

        return disparity_scales


if __name__=='__main__':
    import torch
    x = torch.rand(10,3,512,512)
    model = HDispNet(320)
    output = model(x,x)
    for o in output:

        print(o.size())