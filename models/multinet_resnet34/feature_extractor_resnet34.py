import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34

class FeatureExtractor34(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        donor = resnet34(pretrained=pretrained)
        children = list(donor.children())

        self.layer_0 = nn.Sequential(*children[:3])
        self.max_pool_0 = children[3]
        self.layer_1 = children[4]
        self.layer_2 = children[5]
        self.layer_3 = children[6]
        self.layer_4 = children[7]

    def forward(self, x):
        x0 = self.layer_0(x)
        x = self.max_pool_0(x0)
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1) 
        x3 = self.layer_3(x2) 
        x4 = self.layer_4(x3)      
        return x0, x1, x2, x3, x4


if __name__=='__main__':
    x =torch.rand(10,3,512,512)
    scales = FeatureExtractor34()(x)
    for scale in scales:
        print(scale.size())


