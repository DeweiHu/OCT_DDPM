# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:02:42 2021

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\VI,NF,DDPM\\')
sys.path.insert(0,'E:\\tools\\')

import util
#from DDPM_cifar_dataloader import load_data
from OCT_dataloader import load_train_data
from DDPM_Net import Model

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import random

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def noise_estimation_loss(model,
                          x0:torch.Tensor, 
                          t:torch.LongTensor,
                          e:torch.Tensor,
                          b:torch.Tensor,
                          keepdim=False):
    a = torch.cumprod(1-b,dim=0).index_select(0, t).view(-1, 1, 1, 1).to(device)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def model_predict(x0, betas, t, *, status='train', transpose=True):
    with torch.no_grad():
        a = torch.cumprod(1-betas,dim=0)
        a_t = a[t].to(device)
        e = torch.randn_like(x0).to(device)
        
        if status == 'train':
            xt = torch.sqrt(1-a_t)*e+torch.sqrt(a_t)*x0
        elif status == 'test':
            xt = x0
        else:
            print('status unspecified.')
        
        pred_e = model(xt, torch.from_numpy(np.array([T])).to(device))
        x0_pred = 1/torch.sqrt(a_t)*xt-torch.sqrt(1-a_t)/torch.sqrt(a_t)*pred_e
        
        if transpose:
            x0_show = np.transpose(x0[0,0,:,:].detach().cpu().numpy())
            x0_pred_show = np.transpose(x0_pred[0,0,:,:].detach().cpu().numpy())
            xt_show = np.transpose(xt[0,0,:,:].detach().cpu().numpy())
            
        else:
            x0_show = x0[0,0,:,:].detach().cpu().numpy()
            x0_pred_show = x0_pred[0,0,:,:].detach().cpu().numpy()
            xt_show = xt[0,0,:,:].detach().cpu().numpy()

    return x0_show, x0_pred_show, xt_show

    
#%%
global T
T = 100
gpu = 1
n_epoch = 500

model_dir = 'E:\\Model\\'
device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")

#train_data = load_data(data_dir=data_dir,batch_size=10,image_size=32,class_cond=True)
train_data = load_train_data(batch_size=2)
beta_schedule = get_beta_schedule('linear',
                                  beta_start=0.0001,
                                  beta_end=0.003,
                                  num_diffusion_timesteps=T)
plt.plot(beta_schedule)
beta_schedule = torch.from_numpy(beta_schedule).float().to(device)

model = Model().to(device)
#model.load_state_dict(torch.load(model_dir+'DDPM_oct_dataset2_2021-07-14.pt'))

lr = 1e-4
optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.5,0.999))
scheduler = StepLR(optimizer,step_size=5,gamma=0.5)
    
#%%
for epoch in range(n_epoch):
    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:
        for step, x in enumerate(train_data):
            
            model.train()
            
            x = x.to(device)
            e = torch.randn_like(x).to(device)
            b = beta_schedule
            
            n = x.size(0)  # batch_size
            t = torch.randint(low=0, high=T, size=(n//2 + 1,))
            t = torch.cat([t,T-t-1], dim=0)[:n].to(device)
            
            loss = noise_estimation_loss(model, x, t, e, b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            pbar.set_description('Epoch: %d. Loss value: %.4f' % (epoch+1,loss))
            
            if step % (len(train_data)-1) == 0 and step != 0:
                timestep = random.randint(1,T-1)
                x0,x_pred,xt = model_predict(x,b,timestep,status='train',transpose=False)
                
                plt.figure(figsize=(15,6))
                plt.subplot(1,3,1),plt.imshow(x0[:,:500],cmap='gray'),plt.axis('off'),plt.title('noisy')
                plt.subplot(1,3,2),plt.imshow(x_pred[:,:500],cmap='gray'),plt.axis('off'),plt.title('sample')
                plt.subplot(1,3,3),plt.imshow(xt[:,:500],cmap='gray'),plt.axis('off'),plt.title('t={}'.format(timestep))
                plt.show()
                
        if epoch % 100 == 0 and epoch != 0:
            name = 'DDPM_oct_dataset2_gt=sf.pt'
            torch.save(model.state_dict(),model_dir+name)
    scheduler.step()

#name = 'DDPM_oct_dataset2_{}.pt'.format(str(datetime.date.today()))
name = 'DDPM_oct_dataset2_gt=sf.pt'
torch.save(model.state_dict(),model_dir+name)
