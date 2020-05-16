import os
import torch
import random
import argparse
import numpy as np
import cv2

from package_core.generic_train_test import *
from dataloader import *
from model_unroll import *
from package_core.metrics import *
from package_core.flow_utils import *

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

parser.add_argument('--model_label', type=str, default='pretrained', help='label used to load pre-trained model')

parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--dataset_root_dir', type=str, required=True, help='absolute path for training dataset')
parser.add_argument('--log_dir', type=str, required=True, help='directory used to store trained networks')
parser.add_argument('--results_dir', type=str, required=True, help='directory used to store trained networks')

parser.add_argument('--net_type', type=str, required=True)
parser.add_argument('--pred_mid_gs', action='store_true')
parser.add_argument('--load_gt_flow', action='store_true')
parser.add_argument('--visualize_results', action='store_true')
parser.add_argument('--compute_metrics', action='store_true')

opts=parser.parse_args()

if opts.net_type=='netMiddle':
    opts.pred_mid_gs=True
    
##===================================================##
##*************** Create dataloader *****************##
##===================================================##
dataloader = Create_dataloader(opts)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelUnroll(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Inference(Generic_train_test):
    def augment_data(self, _input):
        im_rs, im_gs, flow = _input

        if flow is not None:
            flow = flow[:,-2:,:,:].clone()    

        # extract ground truth I_gs
        im_gs = im_gs[:,-self.opts.n_chan:,:,:].clone()
        return [im_rs, im_gs, flow, 0]

    def decode_input(self, data):
        im_rs=data['I_rs']
        im_gs=data['I_gs']
        
        flow=None
        mask=None
        if self.opts.load_gt_flow:
            flow=data['flow']
        if self.opts.dataset_type=='Carla':
            mask=data['mask']
            mask=mask[:,-1:,:,:].clone()

        _input = [im_rs, im_gs, flow]
        return self.augment_data(_input), mask

    def test(self):
        sum_psnr=0.
        sum_psnr_mask=0.
        sum_ssim=0.
        sum_time=0.
        f_metric_all=None
        f_metric_avg=None
        n_frames=0
        dir_results=os.path.join(self.opts.results_dir)
        
        if self.opts.compute_metrics and not os.path.exists(dir_results):
            os.makedirs(dir_results)

        if self.opts.compute_metrics:
            f_metric_all=open(os.path.join(dir_results, 'metric_all'), 'w')
            f_metric_avg=open(os.path.join(dir_results, 'metric_avg'), 'w')

            f_metric_all.write('# frame_id, PSNR_pred, PSNR_pred_mask, SSIM_pred, time (seconds)\n')
            f_metric_avg.write('# avg_PSNR_pred, avg_PSNR_pred_mask, avg_SSIM_pred, time (seconds)\n')

        for i, data in enumerate(self.dataloader):
            _input, mask=self.decode_input(data)
            self.model.set_input(_input)
            with torch.no_grad():
                pred_im, pred_flow, pred_mask=self.model.forward()
            
            if self.opts.visualize_results:
                cv2.imshow('im_rs', self.model.im_rs.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,-3:])
                cv2.imshow('im_gs', self.model.im_gs.detach().cpu().numpy().transpose(0,2,3,1)[0])
                cv2.imshow('pred_im', pred_im[0].detach().cpu().numpy().transpose(0,2,3,1)[0])
                cv2.imshow('err_rs', torch.abs(self.model.im_gs - self.model.im_rs[:,-self.opts.n_chan:,:,:]).detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,-3:])
                cv2.imshow('err_pred', torch.abs(self.model.im_gs - pred_im[0]).detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,-3:])
                
                flow = flow_to_numpy_rgb(pred_flow[0])[0]
                cv2.imshow('pred_disp', flow)
                
                cv2.waitKey(0)

            # compute metrics 
            if self.opts.compute_metrics:
                psnr_pred=PSNR(pred_im[0], self.model.im_gs)
                psnr_pred_mask=PSNR(pred_im[0], self.model.im_gs, mask)
                ssim_pred=SSIM(pred_im[0], self.model.im_gs)

                diff_time = 0

                sum_psnr += psnr_pred
                sum_psnr_mask += psnr_pred_mask
                sum_ssim += ssim_pred
                sum_time += diff_time
                n_frames += 1

                print('PSNR(%.2f dB) PSNR_mask(%.2f dB) SSIM(%.2f) time(%.2f seconds)\n' % (psnr_pred, psnr_pred_mask, ssim_pred, diff_time))
                f_metric_all.write('%d %.2f %.2f %.2f %.2f\n' % (i, psnr_pred, psnr_pred_mask, ssim_pred, diff_time))

        if self.opts.compute_metrics:
            psnr_avg = sum_psnr / n_frames
            psnr_avg_mask = sum_psnr_mask / n_frames
            ssim_avg = sum_ssim / n_frames
            time_avg = sum_time / n_frames

            print('PSNR_avg (%.2f dB) PSNR_avg_mask (%.2f dB) SSIM_avg (%.2f) time_avg(%.2f seconds)' % (psnr_avg, psnr_avg_mask, ssim_avg, time_avg))
            f_metric_avg.write('%.2f %.2f %.2f %.2f\n' % (psnr_avg, psnr_avg_mask, ssim_avg, time_avg))

            f_metric_all.close()
            f_metric_avg.close()

Inference(model, opts, dataloader, None).test()


