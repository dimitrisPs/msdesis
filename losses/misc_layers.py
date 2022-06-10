import torch
import torch.nn as nn
import torch.nn.functional as F

# cost volume
class CostVolume(nn.Module):
    def __init__(self, disparity_lvls):
        super().__init__()
        self.lvls = disparity_lvls

    def forward(self, features_1, features_2):
        assert features_1.size() == features_2.size()
        assert len(features_1.size()) == 4
        bs, _, h, w = features_1.size()
        
        cost_vol = features_1.new_zeros((bs, self.lvls, h, w))
        cost_vol[:, 0, :, :] = self.corr(features_1, features_2).squeeze(1)
        for i in range(1, self.lvls):
            cost_vol[:, i, :, i:] =  self.corr(features_1[:,:,:,i:], features_2[:,:,:,:-i]).squeeze(1)
        return cost_vol
    
    def corr(self, f1:torch.Tensor, f2:torch.Tensor):
        return (f1*f2).sum(axis=1, keepdim=True)

class DisparityWarp(nn.Module):

    def forward(self, src: torch.Tensor, disparity: torch.Tensor):
         
        bs, f, h, w = src.size() 
        y_grid = torch.arange(0, h).view(-1 ,1).repeat(1 ,w)
        y_grid = y_grid.view(1 ,1 ,h ,w).repeat(bs ,1 ,1 ,1).to(src.device)
        
        x_grid = torch.arange(0, w).view(1 ,-1).repeat(h ,1)
        x_grid = x_grid.view(1 ,1 ,h ,w).repeat(bs ,1 ,1 ,1).to(src.device)
        
        map_xy = torch.cat((x_grid - disparity,y_grid) ,1)

        map_xy[: ,0 ,: ,:] = 2.0 *map_xy[: ,0 ,: ,:].clone()/(w) -1.0
        map_xy[: ,1 ,: ,:] = 2.0 *map_xy[: ,1 ,: ,:].clone()/(h) -1.0   
        

        map_xy = map_xy.permute(0 ,2 ,3 ,1).to(src.dtype)


        return F.grid_sample(src, map_xy, align_corners=True)


def denormalize(image:torch.Tensor, stats):
    with torch.no_grad():
        denorm_image = image.detach().clone()
        for ch, m, s in zip(range(3), stats['mean'], stats['std']):
            denorm_image[:,ch].mul_(s).add_(m)
    return denorm_image



def disparity_warp(rim, disp):
    """
    warp stereo image (right image) with disparity
    rim: [B, C, H, W] image/tensor
    disp: [B, 1, H, W] (left) disparity
    for disparity (left), we have
        left_image(x,y) = right_image(x-d(x,y),y)
    """
    B, C, H, W = rim.size()
    # mesh grid 
    xx = torch.linspace(0, W - 1, W, device=rim.device, dtype=rim.dtype).repeat(H,1)
    yy = torch.linspace(0, H - 1, H, device=rim.device, dtype=rim.dtype).reshape(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx,yy),1)

    vgrid = grid

    vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-disp[:,0,:,:])/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    return nn.functional.grid_sample(rim, vgrid.permute(0,2,3,1), align_corners=True)



def disparity_warp_amp_mem_leak(src: torch.Tensor, disparity: torch.Tensor):

    assert src.device == disparity.device
    
    bs, f, h, w = src.size()
    # similar to how it's done in opencv. Create create one map for x and
    # one for y. Add the disparity to x map and combine them in the format
    # grid_sample expects. 
    # probably we can move those grids in the init phase and save some time. 
    y_grid = torch.arange(0, h).view(-1 ,1).repeat(1 ,w)
    y_grid = y_grid.view(1 ,1 ,h ,w).repeat(bs ,1 ,1 ,1).to(src.device)
    
    x_grid = torch.arange(0, w).view(1 ,-1).repeat(h ,1)
    x_grid = x_grid.view(1 ,1 ,h ,w).repeat(bs ,1 ,1 ,1).to(src.device)
    
    map_xy = torch.cat((x_grid - disparity,y_grid) ,1).float()
    # scale grid to [-1,1] see pytorch documentation 
    # https://pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.grid_sample
    map_xy[: ,0 ,: ,:] = 2.0 *map_xy[: ,0 ,: ,:].clone()/(w) -1.0
    map_xy[: ,1 ,: ,:] = 2.0 *map_xy[: ,1 ,: ,:].clone()/(h) -1.0   
    
    # map_xy[: ,0 ,: ,:] -= disparity.squeeze(1)

    map_xy = map_xy.permute(0 ,2 ,3 ,1)
    mask = (map_xy[:,:,:,0]>=-1)
    
    mask = mask.unsqueeze(1).repeat(1,f,1,1)

    return F.grid_sample(src, map_xy, align_corners=True), mask

def construct_exclusive_region_mask(estimated_disparity):
    B, C, H, W = estimated_disparity.size()
    # mesh grid 
    grid_0 = torch.linspace(0, W - 1, W, device=estimated_disparity.device, dtype=estimated_disparity.dtype).repeat(H,1)
    grid_0 = grid_0.view(1,1,H,W).repeat(B,3,1,1)
    exclusive_regions = grid_0-estimated_disparity<=0
    return (~exclusive_regions).detach()




