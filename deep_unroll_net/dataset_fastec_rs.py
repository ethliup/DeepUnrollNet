from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage import io

class Dataset_fastec_rs(Dataset):
    def __init__(self, root_dir, seq_len=2, load_middle_gs=False):
        self.I_gs = []
        self.I_rs = []
        self.seq_len = seq_len
        
        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname != 'meta.log':
                    continue   

                # read in seq of images            
                for i in range(34):
                    if not os.path.exists(os.path.join(seq_path, str(i).zfill(3)+'_rolling.png')):
                        continue

                    seq_Irs=[]
                    seq_Igs=[]

                    seq_Irs.append(os.path.join(seq_path, str(i).zfill(3)+'_rolling.png'))
                    if load_middle_gs:
                        seq_Igs.append(os.path.join(seq_path, str(i).zfill(3)+'_global_middle.png'))
                    else:
                        seq_Igs.append(os.path.join(seq_path, str(i).zfill(3)+'_global_first.png'))

                    for j in range(1,seq_len):
                        seq_Irs.append(os.path.join(seq_path, str(i+j).zfill(3)+'_rolling.png'))
                        if load_middle_gs:
                            seq_Igs.append(os.path.join(seq_path, str(i+j).zfill(3)+'_global_middle.png'))
                        else:
                            seq_Igs.append(os.path.join(seq_path, str(i+j).zfill(3)+'_global_first.png'))
                    
                    if not os.path.exists(seq_Irs[-1]):
                        break

                    self.I_rs.append(seq_Irs.copy())
                    self.I_gs.append(seq_Igs.copy())
                    
    def __len__(self):
        return len(self.I_gs)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]

        temp = io.imread(path_rs[0])
        H,W,C=temp.shape
        if C>3:
            C=3

        I_rs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        I_gs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        
        for i in range(self.seq_len):
            I_rs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_rs[i]).transpose(2,0,1)).float()[:3]/255.
            I_gs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_gs[i]).transpose(2,0,1)).float()[:3]/255.
            
        return {'I_rs': I_rs,
                'I_gs': I_gs,
                'path': path_rs}

