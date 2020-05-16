import sys
import math
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from net_unroll import *

class ModelUnroll(ModelBase): 
    def __init__(self, opts):
        super(ModelUnroll, self).__init__()
        self.opts = opts

        if opts.net_type=='netMiddle':
            self.net_G = Net_flow_unroll(opts.n_chan,
                                        opts.n_init_feat,
                                        opts.n_chan,
                                        True,
                                        opts.seq_len,
                                        est_vel=True,
                                        share_encoder=True,
                                        md=[4,4,4],
                                        pred_middle_gs=True).cuda()
        else:
            raise NotImplementedError()

        self.print_networks(self.net_G)

        if self.opts.is_training:
            # create optimizer
            self.optimizer_G = torch.optim.Adam([{'params': self.net_G.parameters()},], lr=opts.lr)            
            self.optimizer_names = ['G']
            self.build_lr_scheduler()

            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            #self.loss_fn_tv2 = EdgeAwareVariationLoss(in1_nc=2, in2_nc=3)
            self.loss_fn_tv2 = VariationLoss(nc=2)

            self.downsample2 = nn.AvgPool2d(2, stride=2)

        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)

    def set_input(self, _input):
        im_rs, im_gs, gt_flow, cH = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.gt_flow = gt_flow
        self.cH = cH

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.gt_flow is not None:
            self.gt_flow = self.gt_flow.cuda()

    def forward(self):
        pred_im, pred_mask, pred_flow = self.net_G(self.im_rs)
        return pred_im, pred_mask, pred_flow

    def optimize_parameters(self):
        self.pred_im, self.pred_mask, self.pred_flow = self.forward()
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()

        #===========================================================#
        #                  Prepare ground truth data                #
        #===========================================================#
        self.im_rs_clone=[self.im_rs[:,-self.opts.n_chan:,:,:].clone()]
        self.im_gs_clone=[self.im_gs]
        self.gt_flow_clone=[self.gt_flow]

        self.nlvs = len(self.pred_im)

        for i in range(1,self.nlvs):
            self.im_gs_clone.append(self.downsample2(self.im_gs_clone[-1]))
            self.im_rs_clone.append(self.downsample2(self.im_rs_clone[-1]))
            if self.gt_flow_clone[-1] is not None:
                self.gt_flow_clone.append(self.downsample2(self.gt_flow_clone[-1]) * 0.5)

        #===========================================================#
        #                       Compute losses                      #
        #===========================================================#
        self.syn_im_rs=[None]*self.nlvs
        self.syn_im_mask=[None]*self.nlvs

        for lv in range(self.nlvs):
            if self.pred_im[lv] is None:
                continue

            self.loss_L1 += self.opts.lamda_L1 *\
                            self.loss_fn_L1(self.pred_im[lv], self.im_gs_clone[lv], self.pred_mask[lv], mean=True)

            self.loss_perceptual += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_im[lv], self.im_gs_clone[lv])

            if self.opts.flow_supervision:
                if self.opts.lamda_gt_flow>1e-6:
                    self.loss_flow += self.loss_fn_L1(self.pred_flow[lv], self.gt_flow_clone[lv], mean=True)*self.opts.lamda_gt_flow
                else:
                    self.syn_im_rs[lv], self.syn_im_mask[lv] = warp_image_flow(self.im_gs_clone[lv], self.pred_flow[lv])
                    self.loss_flow += self.opts.lamda_L1*\
                                        self.loss_fn_L1(self.syn_im_rs[lv], self.im_rs_clone[lv], self.syn_im_mask[lv], mean=True)

            if self.pred_flow[lv] is not None and self.opts.lamda_flow_smoothness>1e-6:
                #self.loss_flow_smoothness += self.loss_fn_tv2(self.pred_flow[lv], self.im_rs_clone[lv], mean=True)*self.opts.lamda_flow_smoothness
                self.loss_flow_smoothness += self.loss_fn_tv2(self.pred_flow[lv], mean=True)*self.opts.lamda_flow_smoothness

        # sum them up
        self.loss_G = self.loss_L1 +\
                        self.loss_perceptual +\
                        self.loss_flow +\
                        self.loss_flow_smoothness 

        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step() 

    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_G, 'G', label, self.opts.log_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_G, 'G', label, self.opts.log_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1'] = self.loss_L1.item()
        losses['loss_perceptual'] = self.loss_perceptual.item()
        losses['loss_flow'] = self.loss_flow.item()
        losses['loss_flow_smoothness'] = self.loss_flow_smoothness.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()

        for lv in range(self.nlvs):
            if self.pred_im[lv] is None:
                continue
            output_visuals['im_gs_'+str(lv)] = self.im_gs_clone[lv]
            output_visuals['im_gs_pred_'+str(lv)] = self.pred_im[lv]
            output_visuals['res_im_gs_'+str(lv)] = torch.abs(self.pred_im[lv] - self.im_gs_clone[lv])*5.

            if self.syn_im_rs[lv] is not None:
                output_visuals['syn_im_rs_'+str(lv)] = self.syn_im_rs[lv]
                output_visuals['res_im_rs_'+str(lv)] = torch.abs(self.syn_im_rs[lv] - self.im_rs_clone[lv])*self.syn_im_mask[lv].float()*5.

            if self.pred_flow[lv] is not None:
                output_visuals['flow_pred_'+str(lv)] = torch.from_numpy(flow_to_numpy_rgb(self.pred_flow[lv]).transpose(0,3,1,2)).float()/255.
                output_visuals['mask_'+str(lv)] = self.pred_mask[lv].clone().repeat(1,3,1,1)
            
            if self.gt_flow_clone[0] is not None:
                output_visuals['flow_'+str(lv)] = torch.from_numpy(flow_to_numpy_rgb(self.gt_flow_clone[lv]).transpose(0,3,1,2)).float()/255.

        return output_visuals


