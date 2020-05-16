import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil

from package_core.generic_train_test import *
from dataloader import *
from model_unroll import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=True, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--n_chan', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--n_init_feat', type=int, default=32, help='number of channels of initial features')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=False)

parser.add_argument('--model_label', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--net_type', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
 
opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelUnroll(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Demo(Generic_train_test):
    def test(self):
        with torch.no_grad():
            seq_lists = os.listdir(self.opts.data_dir)
            for seq in seq_lists:
                im_rs0_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_0.png')
                im_rs1_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_1.png')

                im_rs0 = torch.from_numpy(io.imread(im_rs0_path).transpose(2,0,1))[:3,:448,:].unsqueeze(0).clone()
                im_rs1 = torch.from_numpy(io.imread(im_rs1_path).transpose(2,0,1))[:3,:448,:].unsqueeze(0).clone()

                im_rs = torch.cat([im_rs0,im_rs1], dim=1).float()/255.
                _input = [im_rs, None, None, 0]
                
                self.model.set_input(_input)
                pred_im, _, _=self.model.forward()

                # save results
                im_gs = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1.png'))
                im_rs = io.imread(im_rs1_path)

                im_gs = im_gs[:448].copy()
                im_rs = im_rs[:448].copy()

                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_1.png'), im_gs)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_1.png'), (pred_im[0].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                print('saved', self.opts.results_dir, seq+'_pred_1.png')
                    
Demo(model, opts, None, None).test()


