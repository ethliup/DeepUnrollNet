import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from package_core.net_basics import *
from package_core.utils import *
from forward_warp_package import *
from correlation_package import Correlation

class Pred_image(nn.Module):
    def __init__(self, nc_in, nc_out):
        super(Pred_image, self).__init__()
        self.conv1 = nn.Conv2d(nc_in, nc_out, kernel_size=3, stride=1, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        return x

class Net_vanilla(nn.Module):
    def __init__(self, n_in, n_init, n_out):
        super(Net_vanilla, self).__init__()

        # encoder 
        self.en_conv0 = Conv2d(n_in=n_in, 
                                n_out=n_init, 
                                bn=False, 
                                act_fn=nn.ReLU(), 
                                ker_sz=7, 
                                strd=1)
        self.en_resB0 = Cascaded_resnet_blocks(n_init, 3)

        self.en_conv1 = Conv2d(n_in=n_init, 
                                n_out=n_init*2, 
                                bn=False, 
                                act_fn=nn.ReLU(), 
                                ker_sz=3, 
                                strd=2)
        self.en_resB1 = Cascaded_resnet_blocks(n_init*2, 3)

        self.en_conv2 = Conv2d(n_in=n_init*2, 
                                n_out=n_init*4, 
                                bn=False, 
                                act_fn=nn.ReLU(), 
                                ker_sz=3, 
                                strd=2)
        self.en_resB2 = Cascaded_resnet_blocks(n_init*4, 3)  

        # decoder 
        self.dec_resB2 = Cascaded_resnet_blocks(n_init*4, 3)
        self.dec_upconv2 = Deconv2d(n_init*4, n_out)
        self.pred_im2 = Pred_image(n_init*4, n_out) 

        self.dec_resB1 = Cascaded_resnet_blocks(n_init*2+n_out*2, 3)
        self.dec_upconv1 = Deconv2d(n_init*2+n_out*2, n_out)
        self.pred_im1 = Pred_image(n_init*2+n_out*2, n_out) 

        self.dec_resB0 = Cascaded_resnet_blocks(n_init+n_out*2, 3)
        self.pred_im0 = Pred_image(n_init+n_out*2, n_out)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, im_rs, cH):
        en_x0 = self.en_resB0(self.en_conv0(im_rs))
        en_x1 = self.en_resB1(self.en_conv1(en_x0))
        en_x2 = self.en_resB2(self.en_conv2(en_x1))

        de_x2 = self.dec_resB2(en_x2)
        de_up2 = self.dec_upconv2(de_x2)
        pred_im2 = self.pred_im2(de_x2)

        pred_im2_up = self.upsample2(pred_im2)
        de_x1 = self.dec_resB1(torch.cat([en_x1, de_up2, pred_im2_up], dim=1))
        de_up1 = self.dec_upconv1(de_x1)
        pred_im1 = self.pred_im1(de_x1)

        pred_im1_up = self.upsample2(pred_im1)
        de_x0 = self.dec_resB0(torch.cat([en_x0, de_up1, pred_im1_up], dim=1))
        pred_im0 = self.pred_im0(de_x0)

        return [pred_im0,pred_im1,pred_im2], [None,None,None], [None,None,None]

class Net_unroll(nn.Module):
    def __init__(self, n_in, n_init, n_out, md=4, est_vel=True, pred_mid_gs=False):
        super(Net_unroll, self).__init__()

        self.n_in = n_in
        self.est_vel=est_vel
        self.pred_mid_gs=pred_mid_gs

        #================================================#
        #                    encoder                     #  
        #================================================#
        self.en_conv0 = Conv2d(n_in=n_in, 
                                n_out=n_init, 
                                bn=False, 
                                act_fn=nn.ReLU(), 
                                ker_sz=7, 
                                strd=1)
        self.en_resB0 = Cascaded_resnet_blocks(n_init, 3)

        self.en_conv1 = Conv2d(n_in=n_init, 
                                n_out=n_init*2, 
                                bn=False, 
                                act_fn=nn.ReLU(), 
                                ker_sz=3, 
                                strd=2)
        self.en_resB1 = Cascaded_resnet_blocks(n_init*2, 3)

        self.en_conv2 = Conv2d(n_in=n_init*2, 
                                n_out=n_init*4, 
                                bn=False, 
                                act_fn=nn.ReLU(), 
                                ker_sz=3, 
                                strd=2)
        self.en_resB2 = Cascaded_resnet_blocks(n_init*4, 3)  

        #================================================#
        #                    decoder                     #  
        #================================================#
        self.dec_resB2 = Cascaded_resnet_blocks(n_init*4, 3)
        self.pred_im2 = Pred_image(n_init*4, n_out) 

        self.dec_upconv2 = Deconv2d(n_init*4, n_out)
        self.dec_resB1 = Cascaded_resnet_blocks(n_init*2+n_out*2, 3)
        self.pred_im1 = Pred_image(n_init*2+n_out*2, n_out) 

        self.dec_upconv1 = Deconv2d(n_init*2+n_out*2, n_out)
        self.dec_resB0 = Cascaded_resnet_blocks(n_init+n_out*2, 3)
        self.pred_im0 = Pred_image(n_init+n_out*2, n_out)

        #================================================#
        #                  motion estimator              #  
        #================================================#
        self.corr2    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU2 = nn.LeakyReLU(0.1)

        self.corr1    = Correlation(pad_size=md*2, kernel_size=1, max_displacement=md*2, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU1 = nn.LeakyReLU(0.1)

        # level 2
        dd = np.cumsum([128,128,96,64,32])        
        od = (2*md+1)**2
        self.conv2_0 = Conv2d(od,      128, bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv2_1 = Conv2d(od+dd[0],128, bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv2_2 = Conv2d(od+dd[1],96,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv2_3 = Conv2d(od+dd[2],64,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv2_4 = Conv2d(od+dd[3],32,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.motion2 = Conv2d(od+dd[4],2,   bn=False, act_fn=None,      ker_sz=3, strd=1)

        # level 1
        self.upfeat2 = Deconv2d(od+dd[4], 2) 
        
        dd = np.cumsum([64,64,48,32,16])  
        od = (4*md+1)**2+2
        self.conv1_0 = Conv2d(od,      64,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv1_1 = Conv2d(od+dd[0],64,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv1_2 = Conv2d(od+dd[1],48,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv1_3 = Conv2d(od+dd[2],32,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.conv1_4 = Conv2d(od+dd[3],16,  bn=False, act_fn=nn.ReLU(), ker_sz=3, strd=1)
        self.motion1 = Conv2d(od+dd[4],2,   bn=False, act_fn=None,      ker_sz=3, strd=1)

        # 
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, im_rs, cH):
        im0=im_rs[:,0:self.n_in,:,:].clone()
        im1=im_rs[:,self.n_in:self.n_in*2,:,:].clone()

        #====================================================#
        #                       encode                       #
        #====================================================#
        im0_en0=self.en_resB0(self.en_conv0(im0))
        im0_en1=self.en_resB1(self.en_conv1(im0_en0))
        im0_en2=self.en_resB2(self.en_conv2(im0_en1))

        im1_en0=self.en_resB0(self.en_conv0(im1))
        im1_en1=self.en_resB1(self.en_conv1(im1_en0))
        im1_en2=self.en_resB2(self.en_conv2(im1_en1))
        
        #====================================================#
        #                  estimate motion                   #
        #====================================================#
        # level 2
        corr2 = self.leakyRELU2(self.corr2(im1_en2, im0_en2))  
        x = torch.cat((self.conv2_0(corr2), corr2),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        M2 = self.motion2(x)

        # level 1
        up_x = self.upfeat2(x)

        corr1 = self.leakyRELU1(self.corr1(im1_en1, im0_en1))
        x = torch.cat([corr1, up_x], dim=1)
        x = torch.cat((self.conv1_0(x), x), 1)
        x = torch.cat((self.conv1_1(x), x),1)
        x = torch.cat((self.conv1_2(x), x),1)
        x = torch.cat((self.conv1_3(x), x),1)
        x = torch.cat((self.conv1_4(x), x),1)
        M1 = self.motion1(x)

        #====================================================#
        #                   warp features                    #
        #====================================================#
        # warp pyramid level 2
        B,C,H,W=im1_en2.size()
        warper2 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            grid, _ = generate_2D_mesh(H, W)
            grid_rows = grid[1]

            t = grid_rows.unsqueeze(0).unsqueeze(0)+cH/4.

            if self.pred_mid_gs:
                t = t - H//2

            flow2 = M2 * t
            
        else:
            flow2 = M2

        im1_en2_w, mask2 = warper2(im1_en2, flow2)

        # warp pyramid level 1
        B,C,H,W=im1_en1.size()
        warper1 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            grid, _ = generate_2D_mesh(H, W)
            grid_rows = grid[1]

            t = grid_rows.unsqueeze(0).unsqueeze(0)+cH/2.

            if self.pred_mid_gs:
                t = t - H//2

            res_flow1 = M1 * t
            
        else:
            res_flow1 = M1

        flow1 = self.upsample2(flow2)*2. + res_flow1
        im1_en1_w, mask1 = warper1(im1_en1, flow1)

        # warp pyramid level 0
        B,C,H,W=im1_en0.size()
        warper0 = ForwardWarp.create_with_implicit_mesh(B,C,H,W,2,0.5)
        flow0 = self.upsample2(flow1)*2.
        im1_en0_w, mask0 = warper0(im1_en0, flow0)

        #====================================================#
        #                      decoder                       #
        #====================================================#
        # estimate image at pyramid level 2
        im1_dec2 = self.dec_resB2(im1_en2_w)
        pred_im2 = self.pred_im2(im1_dec2)

        # estimate image at pyramid level 1
        up_pred_im2 = self.upsample2(pred_im2)
        up_im1_dec2 = self.dec_upconv2(im1_dec2)
        im1_dec1 = self.dec_resB1(torch.cat([im1_en1_w, up_pred_im2, up_im1_dec2], dim=1))
        pred_im1 = self.pred_im1(im1_dec1)

        # estimate image at pyramid level 0
        up_pred_im1 = self.upsample2(pred_im1)
        up_im1_dec1 = self.dec_upconv1(im1_dec1)
        im1_dec0 = self.dec_resB0(torch.cat([im1_en0_w, up_pred_im1, up_im1_dec1], dim=1))
        pred_im0 = self.pred_im0(im1_dec0)

        return [pred_im0, pred_im1, pred_im2], [flow0, flow1, flow2], [mask0, mask1, mask2]

if __name__ == '__main__':
    net = Net_unroll(3, 32, 3).cuda()
    x = torch.ones([1, 6, 128, 128]).float().cuda()
    im, flow, mask = net(x, 0)
    print(im[0].size())

    net = Net_vanilla(6,32,3).cuda()
    x = torch.ones([1, 6, 128, 128]).float().cuda()
    im, flow, mask = net(x)
    print(im[0].size())
