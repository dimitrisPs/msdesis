import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_0 = Encoder(3,16)
        self.encoder_1 = Encoder(16,32)
        self.encoder_2 = Encoder(32,64)
        self.encoder_3 = Encoder(64,96)
        self.encoder_4 = Encoder(96,128)

    def forward(self, x):
        f1 = self.encoder_0(x)
        f2 = self.encoder_1(f1)
        f3 = self.encoder_2(f2)
        f4 = self.encoder_3(f3)
        f5 = self.encoder_4(f4)
        return f1, f2, f3, f4, f5
        


class Encoder(nn.Module):
    def __init__(self, ch_in, ch_out, bn=False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=2, bias=False),
                                     nn.BatchNorm2d(ch_out),
                                     nn.LeakyReLU(0.1, inplace=True),
                                     nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(ch_out),
                                     nn.LeakyReLU(0.1, inplace=True))
    def forward(self, x):
        return self.encoder(x)
