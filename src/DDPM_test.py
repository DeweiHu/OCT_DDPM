# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:26:31 2021

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\VI,NF,DDPM\\')
sys.path.insert(0,'E:\\tools\\')

import util
from DDPM_GaussianDiffusion import GaussianDiffusion, get_beta_schedule
from OCT_dataloader import load_test_data
from DDPM_Net import Model

import torch
import numpy as np
import matplotlib.pyplot as plt


global T
T = 100
gpu = 1

model_dir = 'E:\\Model\\'
save_dir = 'E:\\HumanData\\'
device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")

betas = get_beta_schedule('linear',
                          beta_start=0.0001,
                          beta_end=0.006,
                          num_diffusion_timesteps=T)

betas = torch.from_numpy(betas).float().to(device)

model = Model().to(device)
model.load_state_dict(torch.load(model_dir+"DDPM_oct_dataset2_2021-07-08.pt"))

#%%
region = "Fovea"
snr = "92"
idx = "1"
test_data = load_test_data(region=region, snr=snr, idx=idx, batch_size=1)
optimized_t = {"101":41,"96":46,"92":50}

#%%
vol_dn = np.zeros([len(test_data),512,500],dtype=np.float32)

for step, x in enumerate(test_data):
    with torch.no_grad():
        sample = GaussianDiffusion(betas=betas,device=device)
        x0 = x.to(device)
        t = 48
        
        eps_t = model(x0, torch.tensor([t]).to(device))
        x0_pred = sample.denoise(x0, eps_t, t)
        
        im_0 = GaussianDiffusion._to_nparray_(x0_pred,transpose=False)
        im_orig = GaussianDiffusion._to_nparray_(x0,transpose=False)
        
        vol_dn[step,:,:] = im_0
        
        if step % 50 == 0:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1),plt.imshow(im_orig,cmap='gray'),plt.axis('off'),plt.title('noisy')
            plt.subplot(1,2,2),plt.imshow(im_0,cmap='gray'),plt.axis('off'),plt.title('sample')
            plt.show()

folder = region+"_SNR_"+snr+"_"+idx
util.nii_saver(np.transpose(vol_dn,[2,1,0]),save_dir+folder+"\\","DDPM_"+folder+".nii.gz")
#        
#%%
vol_t = np.zeros([T,512,500],dtype=np.float32)

for step, x in enumerate(test_data):
    if step == 200:
        with torch.no_grad():
            sample = GaussianDiffusion(betas=betas,device=device)
            x0 = x.to(device)
            for t in range(T):
                eps_t = model(x0, torch.tensor([t]).to(device))
                x0_pred = sample.denoise(x0, eps_t, t)
                im_0 = GaussianDiffusion._to_nparray_(x0_pred,transpose=False)
                vol_t[t,:,:] = im_0
    else:
        pass
    
util.nii_saver(np.transpose(vol_t,[2,1,0]),'E:\\','vol_92_t.nii.gz')
        
