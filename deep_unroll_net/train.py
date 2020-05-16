import os
import json
import torch
import random
import argparse
import numpy as np

from tensorboardX import SummaryWriter

from package_core.generic_train_test import *
from package_core.metrics import *
from dataloader import *
from model_unroll import *

torch.cuda.set_device(1)
torch.manual_seed(0)

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=3, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=False, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size width')
parser.add_argument('--crop_sz_W', type=int, default=256, help='cropped image size width')
parser.add_argument('--is_training', type=bool, default=True, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=100000, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=10000, help='epoch to start lr decay')
parser.add_argument('--n_chan', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--n_init_feat', type=int, default=32, help='number of channels of initial features')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=True)

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--max_epochs', type=int, default=1001)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--model_label', type=str, default='pretrained', help='label used to load pre-trained model')
parser.add_argument('--pred_mid_gs', type=bool, default=False)

parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--dataset_root_dir', type=str, required=True, help='absolute path for training dataset')
parser.add_argument('--log_dir', type=str, required=True, help='directory used to store trained networks')

parser.add_argument('--net_type', type=str, required=True)
parser.add_argument('--load_gt_flow', action='store_true')
parser.add_argument('--flow_supervision', action='store_true')

parser.add_argument('--lamda_perceptual', type=float, default=1)
parser.add_argument('--lamda_L1', type=float, default=10)
parser.add_argument('--lamda_gt_flow', type=float, required=True)
parser.add_argument('--lamda_flow_smoothness', type=float, required=True)

opts=parser.parse_args()

opts.log_dir = opts.log_dir+opts.dataset_type+\
                '_netType'+str(opts.net_type)+\
                '_flowSuper'+str(opts.flow_supervision)+\
                '_lPercep'+str(opts.lamda_perceptual)+\
                '_lL1'+str(opts.lamda_L1)+\
                '_lgtFlow'+str(opts.lamda_gt_flow)+\
                '_lFlowTV'+str(opts.lamda_flow_smoothness)

if opts.net_type=='netMiddle':
    opts.pred_mid_gs=True

if not os.path.exists(opts.log_dir):
    os.makedirs(opts.log_dir)

config_save_path = os.path.join(opts.log_dir, 'opts.txt')
with open(config_save_path, 'w') as f:
    json.dump(opts.__dict__, f, indent=2)

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
dataloader = Create_dataloader(opts)

# update opts
orig_batch_sz = opts.batch_sz
opts.dataset_root_dir = opts.dataset_root_dir.replace('/train', '/val')
opts.batch_sz = 1

dataloader_val = Create_dataloader(opts)

# recover 
opts.dataset_root_dir = opts.dataset_root_dir.replace('/val', '/train')
opts.batch_sz = orig_batch_sz

##===================================================##
##*************** Create datalogger *****************##
##===================================================##
logger = SummaryWriter(opts.log_dir)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelUnroll(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Train(Generic_train_test):
    def augment_data(self, _input):
        im_rs, im_gs, flow = _input

        # do random crop
        _,_,H,W=im_rs.size() 
        cH=random.randint(0, H-self.opts.crop_sz_H)
        cW=random.randint(0, W-self.opts.crop_sz_W)

        im_rs = im_rs[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()
        im_gs = im_gs[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()

        if flow is not None:
            flow = flow[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()
            flow = flow[:,-2:,:,:].clone()

        # extract ground truth I_gs
        im_gs = im_gs[:,-self.opts.n_chan:,:,:].clone()
        return [im_rs, im_gs, flow, cH]

    def decode_input(self, data):
        im_rs=data['I_rs']
        im_gs=data['I_gs']

        flow=None
        if self.opts.load_gt_flow:
            flow=data['flow']
        _input = [im_rs, im_gs, flow]
        return self.augment_data(_input)

    def validation(self, epoch):
        psnr_sum = 0.
        ssim_sum = 0.

        for i, data in enumerate(self.dataloader_val):
            # decode input
            im_rs=data['I_rs']
            im_gs=data['I_gs'][:,-self.opts.n_chan:,:,:].clone().cuda()

            _input = [im_rs, im_gs, None, 0]
            self.model.set_input(_input)

            with torch.no_grad():
                pred_im, _, _ = self.model.forward()

            # compute metrics 
            psnr_sum += PSNR(pred_im[0], im_gs)
            ssim_sum += SSIM(pred_im[0], im_gs)

        psnr_avg = psnr_sum/len(self.dataloader_val)
        ssim_avg = ssim_sum/len(self.dataloader_val)

        self.logger.add_scalar('PSNR_val', psnr_avg, epoch)
        self.logger.add_scalar('SSIM_val', ssim_avg, epoch)

Train(model, opts, dataloader, logger, dataloader_val).train()


