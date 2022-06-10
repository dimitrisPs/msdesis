import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetHead(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        feature_channels=[64,64,128,256,512]

        self.decode_1 = Decode(feature_channels[4], feature_channels[3])
        self.decode_2 = Decode(feature_channels[3], feature_channels[2])
        self.decode_3 = Decode(feature_channels[2], feature_channels[1])
        self.decode_4 = Decode(feature_channels[1], feature_channels[0])

        self.upsample = nn.ConvTranspose2d(feature_channels[0],feature_channels[0],kernel_size=4, stride=2, padding=1)
        
        self.out_conv = nn.Conv2d(feature_channels[0], n_classes, kernel_size=1)        

    def forward(self, feature_scales):
        x = self.decode_1(feature_scales[-1], feature_scales[-2])
        x = self.decode_2(x, feature_scales[-3])
        x = self.decode_3(x, feature_scales[-4])
        x = self.decode_4(x, feature_scales[-5])
        x = self.upsample(x)
        x = self.out_conv(x)
        return x



class ConvBlock(nn.Module):
    
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch_out),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch_out),
                                   nn.LeakyReLU(0.1))
        
    def forward(self, x):

        return self.layer(x)
        

class Decode(nn.Module):
    def __init__(self, ch_in, ch_out, aux_in=0):
        super().__init__()
        self.d_conv = ConvBlock(ch_out*2+aux_in, ch_out)
        self.up_conv = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x, x_c, aux=None):
        # deconv x
        x = self.up_conv(x)
        if aux is not None:
            x = torch.cat((x, aux), dim=1)
        x = torch.cat((x_c, x), dim=1)
        x = self.d_conv(x)
        return x
    