import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from correlation_package import Correlation

from package_core.net_basics import *
from forward_warp_package import *

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 

    return grid

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

class Image_encoder(nn.Module):
    def __init__(self, nc_in, nc_init):
        super(Image_encoder, self).__init__()
        
        self.conv_0 = conv2d(in_planes=nc_in, 
                            out_planes=nc_init, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=7, 
                            stride=1)
        self.resnet_block_0 = Cascade_resnet_blocks(in_planes=nc_init, n_blocks=3)

        self.conv_1 = conv2d(in_planes=nc_init, 
                            out_planes=nc_init*2, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=3, 
                            stride=2)
        self.resnet_block_1 = Cascade_resnet_blocks(in_planes=nc_init*2, n_blocks=3)

        self.conv_2 = conv2d(in_planes=nc_init*2, 
                            out_planes=nc_init*4, 
                            batch_norm=False, 
                            activation=nn.ReLU(), 
                            kernel_size=3, 
                            stride=2)
        self.resnet_block_2 = Cascade_resnet_blocks(in_planes=nc_init*4, n_blocks=3)        
                
    def forward(self, x):
        x0 = self.resnet_block_0(self.conv_0(x))
        x1 = self.resnet_block_1(self.conv_1(x0))
        x2 = self.resnet_block_2(self.conv_2(x1))
        return x2, x1, x0

class Image_decoder(nn.Module):
    def __init__(self, nc_init, nc_out, multi_lvs):
        super(Image_decoder, self).__init__()

        nc=nc_out
        self.multi_lvs=multi_lvs
        if multi_lvs:
            nc=nc_out*2
            self.pred_im2 = Pred_image(nc_in=nc_init*4, nc_out=nc_out) 
            self.pred_im1 = Pred_image(nc_in=nc_init*2+nc, nc_out=nc_out) 

        self.resnet_block_2 = Cascade_resnet_blocks(in_planes=nc_init*4, n_blocks=3)
        self.upconv_2 = deconv2d(in_planes=nc_init*4, out_planes=nc_out)

        self.resnet_block_1 = Cascade_resnet_blocks(in_planes=nc_init*2+nc, n_blocks=3)
        self.upconv_1 = deconv2d(in_planes=nc_init*2+nc, out_planes=nc_out)

        self.resnet_block_0 = Cascade_resnet_blocks(in_planes=nc_init+nc, n_blocks=3)
        self.pred_im0 = Pred_image(nc_in=nc_init+nc, nc_out=nc_out)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x2, x1, x0):
        im = [None, None, None]
        
        # level 2
        x2 = self.resnet_block_2(x2)
        up_x2 = self.upconv_2(x2)
        cat_x2 = torch.cat([x1, up_x2], dim=1)
        if self.multi_lvs:
            im2=self.pred_im2(x2)
            up_im2 = self.upsample2(im2)
            cat_x2 = torch.cat([cat_x2, up_im2], dim=1)
            im[2] = im2
        
        # level 1
        x1 = self.resnet_block_1(cat_x2)
        up_x1 = self.upconv_1(x1)
        cat_x1 = torch.cat([x0, up_x1], dim=1)
        if self.multi_lvs:
            im1 = self.pred_im1(x1)
            up_im1 = self.upsample2(im1)
            cat_x1 = torch.cat([cat_x1, up_im1], dim=1)
            im[1] = im1
        
        # level 0
        x0 = self.resnet_block_0(cat_x1)
        im0 = self.pred_im0(x0)
        im[0] = im0
        
        return im

class Flow_decoder(nn.Module):
    def __init__(self, n_inputs, share_encoder, md=[4, 4, 4]):
        super(Flow_decoder, self).__init__()

        self.share_encoder=share_encoder
        if not share_encoder:
            self.conv0a  = conv2d(3,   16, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
            self.conv0aa = conv2d(16,  16, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
            self.conv0b  = conv2d(16,  16, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
            self.conv1a  = conv2d(16,  32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=2)
            self.conv1aa = conv2d(32,  32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
            self.conv1b  = conv2d(32,  32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
            self.conv2a  = conv2d(32,  64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=2)
            self.conv2aa = conv2d(64,  64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
            self.conv2b  = conv2d(64,  64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        
        self.corr2    = Correlation(pad_size=md[2], kernel_size=1, max_displacement=md[2], stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU2 = nn.LeakyReLU(0.1)

        self.corr1    = Correlation(pad_size=md[1], kernel_size=1, max_displacement=md[1], stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU1 = nn.LeakyReLU(0.1)

        self.corr0    = Correlation(pad_size=md[0], kernel_size=1, max_displacement=md[0], stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU0 = nn.LeakyReLU(0.1)


        dd = np.cumsum([128,128,96,64,32])        
        nd = (2*md[2]+1)**2

        # level 2
        od = nd*(n_inputs-1)
        self.conv2_0 = conv2d(od,      128, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_1 = conv2d(od+dd[0],128, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_2 = conv2d(od+dd[1],96,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_3 = conv2d(od+dd[2],64,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv2_4 = conv2d(od+dd[3],32,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.predict_flow2 = self.predict_flow(od+dd[4]) 
        self.deconv2 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat2 = self.deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 

        # level 1
        dd = np.cumsum([64,64,48,32,16])  
        nd = (2*md[1]+1)**2
        od = nd*(n_inputs-1)+2+2
        self.conv1_0 = conv2d(od,      64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_1 = conv2d(od+dd[0],64, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_2 = conv2d(od+dd[1],48,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_3 = conv2d(od+dd[2],32,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.conv1_4 = conv2d(od+dd[3],16,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        self.predict_flow1 = self.predict_flow(od+dd[4]) 
        self.deconv1 = self.deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat1 = self.deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        self.upsample2 = nn.Upsample(scale_factor=2)

        # # level 0
        # dd = np.cumsum([32,32,24,16,8]) 
        # nd = (2*md[0]+1)**2
        # od = nd*(n_inputs-1)+2+2
        # self.conv0_0 = conv2d(od,      32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        # self.conv0_1 = conv2d(od+dd[0],32, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        # self.conv0_2 = conv2d(od+dd[1],24, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        # self.conv0_3 = conv2d(od+dd[2],16, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        # self.conv0_4 = conv2d(od+dd[3],8,  batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)
        # self.predict_flow0 = self.predict_flow(od+dd[4]) 
        
    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return nn.ConvTranspose2d(int(in_planes), out_planes, kernel_size, stride, padding, bias=True)

    def forward(self, in0, in1, in2=None):
        if not self.share_encoder:
            c00=self.conv0b(self.conv0aa(self.conv0a(in0)))
            c10=self.conv0b(self.conv0aa(self.conv0a(in1)))

            c01=self.conv1b(self.conv1aa(self.conv1a(c00)))
            c11=self.conv1b(self.conv1aa(self.conv1a(c10)))
            
            c02=self.conv2b(self.conv2aa(self.conv2a(c01)))
            c12=self.conv2b(self.conv2aa(self.conv2a(c11)))    

            if in2 is not None:
                c20=self.conv0b(self.conv0aa(self.conv0a(in2)))
                c21=self.conv1b(self.conv1aa(self.conv1a(c20)))
                c22=self.conv2b(self.conv2aa(self.conv2a(c21)))
        else:
            c02,c01,c00 = in0
            c12,c11,c10 = in1
            if in2 is not None:
                c22,c21,c20 = in2

        # level 2
        corr2 = self.leakyRELU2(self.corr2(c12, c02))  
        if in2 is not None:
            corr22 = self.leakyRELU2(self.corr2(c12, c22))  
            corr2 = torch.cat([corr2, corr22], dim=1)

        x = torch.cat((self.conv2_0(corr2), corr2),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
        upflow2 = self.deconv2(flow2)
        upfeat2 = self.upfeat2(x)

        # level 1
        corr1 = self.leakyRELU1(self.corr1(c11, c01))  
        if in2 is not None:
            corr21 = self.leakyRELU1(self.corr1(c11, c21))  
            corr1 = torch.cat([corr1, corr21], dim=1)

        x = torch.cat([corr1, upfeat2, upflow2], dim=1)
        x = torch.cat((self.conv1_0(x), x),1)
        x = torch.cat((self.conv1_1(x), x),1)
        x = torch.cat((self.conv1_2(x), x),1)
        x = torch.cat((self.conv1_3(x), x),1)
        x = torch.cat((self.conv1_4(x), x),1)
        flow1 = self.predict_flow1(x) + upflow2*2.0
        upflow1 = self.deconv1(flow1)
        upfeat1 = self.upfeat1(x)
        
        upflow1 = self.upsample2(flow1)*0.5

        # # level 0
        # corr0 = self.leakyRELU0(self.corr0(c10, c00))
        # if in2 is not None:  
        #     corr20 = self.leakyRELU0(self.corr0(c10, c20))  
        #     corr0 = torch.cat([corr0, corr20], dim=1)

        # x = torch.cat([corr0, upfeat1, upflow1], dim=1)
        # x = torch.cat((self.conv0_0(x), x),1)
        # x = torch.cat((self.conv0_1(x), x),1)
        # x = torch.cat((self.conv0_2(x), x),1)
        # x = torch.cat((self.conv0_3(x), x),1)
        # x = torch.cat((self.conv0_4(x), x),1)
        # flow0 = self.predict_flow0(x) + upflow1*2.0
        flow0 = upflow1*2.0
        
        return flow0, flow1, flow2

class Net_vanilla_auto_encoder(nn.Module):
    def __init__(self, nc_in, nc_init, nc_out, multi_lvs, n_inputs):
        super(Net_vanilla_auto_encoder, self).__init__()

        self.encoder = Image_encoder(nc_in*n_inputs, nc_init)
        self.decoder = Image_decoder(nc_init, nc_out, multi_lvs)

    def forward(self, im_rs):
        x2,x1,x0 = self.encoder(im_rs)
        im = self.decoder(x2,x1,x0)
        return im, None, None

class Net_flow_unroll(nn.Module):
    def __init__(self, nc_in, nc_init, nc_out, multi_lvs, n_inputs, est_vel, share_encoder, md, pred_middle_gs):
        super(Net_flow_unroll, self).__init__()
        self.nc_in = nc_in
        self.n_inputs = n_inputs
        self.share_encoder = share_encoder
        self.est_vel = est_vel
        self.pred_middle_gs = pred_middle_gs

        self.encoder = Image_encoder(nc_in, nc_init)
        self.flow_decoder = Flow_decoder(n_inputs, share_encoder, md)
        self.im_decoder = Image_decoder(nc_init, nc_out, multi_lvs)

    def forward(self, im_rs):
        im_rs0 = im_rs[:,0:self.nc_in,:,:].clone()
        im_rs1 = im_rs[:,self.nc_in:self.nc_in*2,:,:].clone()
        im_rs2 = None
        if self.n_inputs>2:
            im_rs2 = im_rs[:,self.nc_in*2:self.nc_in*3,:,:].clone()

        # encoder 
        feat_im0 = self.encoder(im_rs0)
        feat_im1 = self.encoder(im_rs1)
        feat_im2 = None
        if self.n_inputs>2:
            feat_im2 = self.encoder(im_rs2)

        # motion decoder 
        if self.share_encoder:
            flow0, flow1, flow2 = self.flow_decoder(feat_im0, feat_im1, feat_im2)
        else:
            flow0, flow1, flow2 = self.flow_decoder(im_rs0, im_rs1, im_rs2)

        # warp image features
        c12, c11, c10 = feat_im1

        B,C,H,W=c12.size()
        warper2 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            grid_rows=generate_2D_grid(H, W)[1]
            t_flow_ref_to_row0=grid_rows.unsqueeze(0).unsqueeze(0)
            if self.pred_middle_gs:
                t_flow_ref_to_row0=t_flow_ref_to_row0-H//2
            flow2=flow2*t_flow_ref_to_row0
        c12_warped, mask2 = warper2(c12, flow2)

        B,C,H,W=c11.size()
        warper1 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            grid_rows=generate_2D_grid(H, W)[1]
            t_flow_ref_to_row0=grid_rows.unsqueeze(0).unsqueeze(0)
            if self.pred_middle_gs:
                t_flow_ref_to_row0=t_flow_ref_to_row0-H//2
            flow1=flow1*t_flow_ref_to_row0
        c11_warped, mask1 = warper1(c11, flow1)

        B,C,H,W=c10.size()
        warper0 = ForwardWarp.create_with_implicit_mesh(B, C, H, W, 2, 0.5)
        if self.est_vel:
            grid_rows=generate_2D_grid(H, W)[1]
            t_flow_ref_to_row0=grid_rows.unsqueeze(0).unsqueeze(0)
            if self.pred_middle_gs:
                t_flow_ref_to_row0=t_flow_ref_to_row0-H//2
            flow0=flow0*t_flow_ref_to_row0
        c10_warped, mask0 = warper0(c10, flow0)

        # image decoder 
        im = self.im_decoder(c12_warped, c11_warped, c10_warped)

        return im, [mask0, mask1, mask2], [flow0, flow1, flow2]

