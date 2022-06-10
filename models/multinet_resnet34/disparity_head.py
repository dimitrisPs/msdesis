import torch 
import torch.nn as nn
import torch.nn.functional as F




class MatchNet(nn.Module):
    def __init__(self, ch_in, max_disparity):
        super().__init__()
        self.cost_volume = CostVolume(max_disparity)
        self.estimation = nn.Sequential(ConvBnLRelu(ch_in+ max_disparity, 128, k=3, p=1),
                                        ConvBnLRelu(128, 128, k=3, p=1),
                                        ConvBnLRelu(128, 96, k=3, p=1),
                                        ConvBnLRelu(96, 64, k=3, p=1),
                                        ConvBnLRelu(64, 32, k=3, p=1),
                                        nn.Conv2d(32,1,3,1,1))

    def forward(self, left_features, right_features):
        cv = self.cost_volume(left_features, right_features)
        x = torch.cat((cv, left_features), dim=1)
        x = self.estimation(x)
        return x

class MatchNet2(nn.Module):
    def __init__(self, ch_in, max_disparity):
        super().__init__()
        self.cost_volume = CostVolumeBidirectional(max_disparity)
        self.estimation = nn.Sequential(ConvBnLRelu(ch_in+ 2*max_disparity+1, 128, k=3, p=1),
                                        ConvBnLRelu(128, 128, k=3, p=1),
                                        ConvBnLRelu(128, 96, k=3, p=1),
                                        ConvBnLRelu(96, 64, k=3, p=1),
                                        ConvBnLRelu(64, 32, k=3, p=1),
                                        nn.Conv2d(32,1,3,1,1))

    def forward(self, left_features, right_features):
        cv = self.cost_volume(left_features, right_features)
        x = torch.cat((cv, left_features), dim=1)
        x = self.estimation(x)
        return x
        

class RefineDisp(nn.Module):
    def __init__(self, ch_in):
        super().__init__()

        self.refine = nn.Sequential(ConvBnLRelu(ch_in+1, 128, k=3, p=1),
                                    ConvBnLRelu(128, 128, k=3, p=1),
                                    ConvBnLRelu(128, 128, k=3, p=1),
                                    ConvBnLRelu(128, 96, k=3, p=1),
                                    ConvBnLRelu(96, 64, k=3, p=1),
                                    ConvBnLRelu(64, 32, k=3, p=1),
                                    nn.Conv2d(32, 1, kernel_size=3, padding=1))

    def forward(self, init_disparity, features):
        identity = init_disparity
        x = self.refine(torch.cat((features, init_disparity), dim=1))
        return init_disparity + identity


class hourglass2D(nn.Module):
    def __init__(self, inplanes):
        super().__init__()

        self.conv1 = ConvBnLRelu(inplanes, inplanes*2, k=3, s=2, p=1)

        self.conv2 = ConvBnLRelu(inplanes*2, inplanes*2, k=3, s=1, p=1)

        self.conv3 = ConvBnLRelu(inplanes*2, inplanes*2, k=3, s=2, p=1)

        self.conv4 = ConvBnLRelu(inplanes*2, inplanes*2, k=3, s=1, p=1)

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes*2, kernel_size=4, padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes, kernel_size=4, padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(inplanes),
                                   nn.LeakyReLU(inplace=True, negative_slope=0.1),
                                   nn.Conv2d(inplanes, 1, kernel_size=3, padding=1, stride=1))

    def forward(self, x):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16


        post = F.leaky_relu(self.conv5(out)+pre,negative_slope=0.1) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, pad=0, stride=1):
        super().__init__()
        self.cbnr= nn.Sequential(nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))
    def forward(self, x):
        return self.cbnr(x)

class TransposeConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, pad=0, stride=1):
        super().__init__()
        self.tcbnr= nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, ks, stride, pad, bias=False),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))
    def forward(self, x):
        return self.tcbnr(x)

class CostVolume(nn.Module):
    def __init__(self, disparity_lvls):
        super().__init__()
        self.lvls = disparity_lvls

    def forward(self, features_1, features_2):
        assert features_1.size() == features_2.size()
        assert len(features_1.size()) == 4
        bs, fs, h, w = features_1.size()
        
        cost_vol = features_1.new_zeros((bs, self.lvls, h, w))
        cost_vol[:, 0, :, :] = self.corr(features_1, features_2)
        for i in range(1, self.lvls):
            cost_vol[:, i, :, i:] =  self.corr(features_1[:,:,:,i:], features_2[:,:,:,:-i])
        return cost_vol
    
    def corr(self, f1:torch.Tensor, f2:torch.Tensor):
        return (f1*f2).mean(axis=1, keepdim=False)


class CostVolumeBidirectional(nn.Module):
    def __init__(self, disparity_lvls):
        super().__init__()
        self.lvls = disparity_lvls

    def forward(self, features_1, features_2):
        assert features_1.size() == features_2.size()
        assert len(features_1.size()) == 4
        bs, fs, h, w = features_1.size()
        
        cost_vol = features_1.new_zeros((bs, 1+2*self.lvls, h, w))

        for i in range(-self.lvls, self.lvls+1):

            cost_vol[:,self.lvls+i,:, max(i,0):min(w, w+i)] = self.corr(features_1[:,:,:,max(i,0):min(w, w+i)],
            features_2[:,:,:,max(-i,0):min(w, w-i)])
        return cost_vol


    def corr(self, f1:torch.Tensor, f2:torch.Tensor):
        return (f1*f2).mean(axis=1, keepdim=False)

class DisparityWarp(nn.Module):

    def forward(self, src: torch.Tensor, disparity: torch.Tensor):
         
        B, C, H, W = src.size()
        # mesh grid 
        xx = torch.linspace(0, W - 1, W, device=src.device, dtype=src.dtype).repeat(H,1)
        yy = torch.linspace(0, H - 1, H, device=src.device, dtype=src.dtype).reshape(-1,1).repeat(1,W)

        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)

        grid = torch.cat((xx,yy),1)

        vgrid = grid

        vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-disparity[:,0,:,:])/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        return nn.functional.grid_sample(src, vgrid.permute(0,2,3,1), align_corners=False)



class DispHead(nn.Module):
    def __init__(self, max_disparity):
        super().__init__()
        self.disp_decoder_5 = MatchNet(512, (max_disparity//32)+1)
        self.disp_decoder_4 = MatchNet2(256, 2)
        self.disp_decoder_3 = MatchNet2(128, 2)
        self.disp_decoder_2 = MatchNet2(64, 2)
        self.disp_decoder_1 = MatchNet2(64, 2)
        self.disp_warp = DisparityWarp()

        self.upsample5 =Upscale(2)
        self.upsample4 =Upscale(2)
        self.upsample3 =Upscale(2)
        self.upsample2 =Upscale(2)
        self.upsample1 =Upscale(2)

        # self.refine = RefineDisp(3)
        self.refine = hourglass2D(4)


    def forward(self, f_scales_l, f_scales_r):
        #for every scale build a cost volume


        d5 = self.disp_decoder_5(f_scales_l[5], f_scales_r[5])


        d4 = self.upsample5(d5)
        f_warped = self.disp_warp(f_scales_r[4], d4)
        d4 = self.disp_decoder_4(f_scales_l[4], f_warped)+d4


        d3 = self.upsample4(d4)
        f_warped = self.disp_warp(f_scales_r[3], d3)
        d3 = self.disp_decoder_3(f_scales_l[3], f_warped)+d3


        d2 = self.upsample3(d3)
        f_warped = self.disp_warp(f_scales_r[2], d2)
        d2 = self.disp_decoder_2(f_scales_l[2], f_warped)+d2


        d1 =self.upsample2(d2)
        f_warped = self.disp_warp(f_scales_r[1], d1)
        d1 = self.disp_decoder_1(f_scales_l[1], f_warped)+d1


        #upsample without mini cost volume and refine, probably this way we could be able to tackle the instability from rectification ambiguities.
        d0 = self.upsample1(d1)
        d0= self.refine(torch.cat((d0, f_scales_r[0]), dim=1))+d0


        return d0, d1, d2, d3, d4, d5


class ConvBnLRelu(nn.Module):
    def __init__(self,ch_in, ch_out, k=3, s=1, p=0, d=1):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(ch_in, ch_out, k, s, p, d, bias=False),
                                nn.BatchNorm2d(ch_out),
                                nn.LeakyReLU(0.1, inplace=True))
    def forward(self, x):
        return self.layer(x)


class Upscale(nn.Module):
    
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(1,1,3,1,1)
    def forward(self, tensor):
        return self.conv(F.interpolate(tensor, scale_factor=self.factor, mode='bilinear', align_corners=False))



if __name__=='__main__':
    x1 = torch.rand(1,32,1,8)*2
    x2 = torch.rand(1,32,1,8)*3
    import time
    cv = CostVolume(3)
    cv_b = CostVolumeBidirectional(2)


    out= cv(x1, x2)
    out_b = cv_b(x1, x2)

    print(out[:,2])
    print(out_b[:,4])



    # print(out_new-out_old)


