# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:58:40 2021

@author: hudew
"""

import os, sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

#%%
class trainDataset(Dataset):
    
    data_dir = "E:\\HumanData\\"
    
    def __init__(self):
        super().__init__()
        self.vol_dir = []
        self.data = []
        
        for folder in os.listdir(trainDataset.data_dir):
            if folder.startswith('ONH'):
                file_name = 'SF_'+folder+'.nii.gz'
                self.vol_dir.append(trainDataset.data_dir+folder+"\\"+file_name)
        
        for i in range(len(self.vol_dir)):
            vol = util.nii_loader(self.vol_dir[i])
            nslc,h,w = vol.shape
            
            for j in range(nslc):
                im = np.zeros([512,512],dtype=np.float32)
                im = util.ImageRescale(vol[j,:,:],[-1,1])
                self.data.append(im)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.tensor(x).type(torch.FloatTensor)
        x = x[None,:,:]
        return x
    
def load_train_data(*, batch_size):
    dataset = trainDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class testDataset(Dataset):
    
    data_dir = "E:\\HumanData\\"
    
    def __init__(self, region, snr, idx):
        super().__init__()
        self.region = region
        self.snr = snr
        self.idx = idx
        self.vol_dir = []
        self.data = []
        
        assert (self.region == 'Fovea' or self.region == 'ONH')
        assert (self.snr == '92' or self.snr == '96' or self.snr == '101')
        
        for folder in os.listdir(testDataset.data_dir):
            vol_reg, _, vol_snr, vol_idx = folder.split('_')
            if self.region == vol_reg and self.snr == vol_snr and self.idx == vol_idx:
                file = "HN_"+folder+".nii.gz"
                self.vol_dir.append(testDataset.data_dir+folder+"\\"+file)
        
        for i in range(len(self.vol_dir)):   
            vol = util.nii_loader(self.vol_dir[i])
            _,nslc,h,w = vol.shape
            
            for j in range(nslc):
                im = np.zeros([512,512],dtype=np.float32)
                im[:,:500] = util.ImageRescale(vol[0,j,:,:],[-1,1])
                self.data.append(im)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.tensor(x).type(torch.FloatTensor)
        x = x[None,:,:]
        return x

def load_test_data(*, region, snr, idx, batch_size, shuffle=False):
    dataset = testDataset(region, snr, idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

#%%
data_dir = "E:\\HumanData\\ONH_SNR_101_1\\"
#
vol = util.nii_loader(data_dir+"SF_ONH_SNR_101_1.nii.gz")     
