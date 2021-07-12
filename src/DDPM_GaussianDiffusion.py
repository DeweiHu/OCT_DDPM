import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\VI,NF,DDPM\\')
sys.path.insert(0,'E:\\tools\\')

import util
import torch
import numpy as np


class GaussianDiffusion:
    
    def __init__(self, *, betas, device):
        self.device = device
        self.betas = betas.to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
    
    def diffuse(self, x0, t):
        assert t <= self.betas.shape[0]
        
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
            
        x0 = torch.tensor(x0).to(self.device)
        eps = torch.randn_like(x0).to(self.device)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        
        xt = torch.sqrt(alpha_cumprod_t)*x0 + \
            torch.sqrt(1-alpha_cumprod_t)*eps
        
        return xt
    
    def denoise(self, xt, eps, t):
        '''
        predict x0 from eps
        '''
        assert xt.shape == eps.shape
        
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
        
        xt = torch.tensor(xt).to(self.device)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        
        # estimate x0 from the predicted epsilon and a estimated t
        x0_pred = 1/torch.sqrt(alpha_cumprod_t)*xt - \
                torch.sqrt(1-alpha_cumprod_t)/torch.sqrt(alpha_cumprod_t)*eps 
        
        return x0_pred
    
    def reverse(self, xt, eps, t):
        '''
        predict x_{t-1} from x_{t} and eps
        '''
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
        
        beta_t = self.betas.index_select(0,t)
        alpha_t = self.alphas.index_select(0,t)
        alpha_cumprod_t = self.alphas_cumprod.index_select(0,t)
        alpha_cumprod_prev = self.alphas_cumprod.index_select(0,t-1)
        
        c1 = torch.sqrt(alpha_cumprod_prev)*beta_t / 1-alpha_cumprod_prev
        c2 = torch.sqrt(alpha_t)*(1-alpha_cumprod_prev) / 1-alpha_cumprod_t
        
        x0_pred = self.denoise(xt, eps, t)
        x_prev = c1*x0_pred + c2*xt
        
        return x_prev
    
    @staticmethod
    def _to_nparray_(x, *, rescale=True, transpose=True):
        x = x[0,0,:,:].detach().cpu().numpy()
        if transpose:
            x = np.transpose(x)
        if rescale:
            x = util.ImageRescale(x,[0,255])
        return x[:,:500]


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
