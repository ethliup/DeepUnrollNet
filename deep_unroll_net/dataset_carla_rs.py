import os
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from package_core.flow_utils import *

class Dataset_carla_rs(Dataset):
    def __init__(self, 
                root_dir, 
                seq_len=2, 
                load_middle_gs=False, 
                load_flow=False, 
                load_mask=False, 
                load_depth=False,
                img_ext='.jpg'):
        self.load_flow=load_flow
        self.load_mask=load_mask
        self.load_depth=load_depth

        self.depth = []
        self.vel = []
        self.I_gs = []
        self.I_rs = []
        self.flow = []
        self.mask = []
        self.seq_len = seq_len
        
        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname != 'gt_vel.log':
                    continue

                # read in ground truth velocity
                path_vel = os.path.join(seq_path, 'gt_vel.log')
                _vel = open(path_vel, 'r').readlines()
                for v in _vel:
                    if '#' in v:
                        continue
                    v = v.replace('\n','')
                    v = v.split(' ')
                    vel = [float(x)*(-1.) for x in v]    

                # read in seq of images
                seq_Irs=[]
                seq_Igs=[]
                seq_Drs=[]
                seq_flow=[]
                seq_mask=[]

                for i in range(10):
                    if not os.path.isfile(os.path.join(seq_path, str(i).zfill(4)+'_rs'+img_ext)):
                        continue

                    seq_Irs.append(os.path.join(seq_path, str(i).zfill(4)+'_rs'+img_ext))
                    if load_middle_gs:
                        seq_Igs.append(os.path.join(seq_path, str(i).zfill(4)+'_gs_m'+img_ext))
                    else:
                        seq_Igs.append(os.path.join(seq_path, str(i).zfill(4)+'_gs_f'+img_ext))
                    seq_Drs.append(os.path.join(seq_path, str(i).zfill(4)+'_rs.pdepth'))
                    seq_flow.append(os.path.join(seq_path, str(i).zfill(4)+'.flow'))
                    seq_mask.append(os.path.join(seq_path, str(i).zfill(4)+'_mask'+img_ext))

                    if not os.path.exists(seq_Irs[-1]):
                        break

                    if len(seq_Irs)<seq_len:
                        continue

                    self.I_rs.append(seq_Irs.copy())
                    self.I_gs.append(seq_Igs.copy())
                    self.depth.append(seq_Drs.copy())
                    self.vel.append(vel)
                    self.flow.append(seq_flow.copy())
                    self.mask.append(seq_mask.copy())

                    seq_Irs.pop(0)
                    seq_Igs.pop(0)
                    seq_Drs.pop(0)
                    seq_flow.pop(0)
                    seq_mask.pop(0)

                # K = np.array([320, 320, 320, 224]).astype(float)
                # for i in range(10):
                #     path_Drs = os.path.join(seq_path, str(i).zfill(4)+'_rs.depth')
                    
                #     if not os.path.exists(path_Drs):
                #         break
                #     if os.path.exists(os.path.join(seq_path, str(i).zfill(4)+'_rs.pdepth')):
                #         continue

                #     D = np.loadtxt(path_Drs)
                #     D = depth_map_radial_to_planar(D, K)
                #     np.savetxt(os.path.join(seq_path, str(i).zfill(4)+'_rs.pdepth'), D, '%.4f')
                #     print('saving', os.path.join(seq_path, str(i).zfill(4)+'_rs.pdepth'))

    def __len__(self):
        return len(self.vel)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]
        path_depth = self.depth[idx]
        vel = torch.tensor(self.vel[idx]).float()
        path_flow = self.flow[idx]
        path_mask = self.mask[idx]

        temp = io.imread(path_rs[0])
        H,W,C=temp.shape
        if C>3:
            C=3

        I_rs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        I_gs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        D_rs=torch.empty([self.seq_len  , H, W], dtype=torch.float32)
        flow=torch.empty([self.seq_len*2, H, W], dtype=torch.float32)
        mask=torch.empty([self.seq_len  , H, W], dtype=torch.float32)
        
        for i in range(self.seq_len):
            I_rs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_rs[i]).transpose(2,0,1)).float()[:3]/255.
            I_gs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_gs[i]).transpose(2,0,1)).float()[:3]/255.
            if self.load_depth:
                D_rs[i:(i+1),:,:] = torch.from_numpy(np.loadtxt(path_depth[i])).unsqueeze(0)
            if self.load_flow:
                flow[i*2:(i+1)*2,:,:] = torch.from_numpy(load_flow(path_flow[i]).transpose(2,0,1)).float()
            if self.load_mask:
                mask[i:(i+1),:,:] = torch.from_numpy(io.imread(path_mask[i])).unsqueeze(0).unsqueeze(0).float()
            
        return {'I_rs': I_rs,
                'I_gs': I_gs,
                'D_rs': D_rs,
                'vel': vel,
                'path': path_rs,
                'flow': flow,
                'mask': mask,
                }

# dataset = Dataset_carla_rs(root_dir='/home/peidong/leonhard/project/infk/cvg/liup/mydata/Unreal/2019_10_20_Carla_RS_dataset/test', seq_len=1)
# print(len(dataset))


