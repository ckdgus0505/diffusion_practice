import numpy as np
import torch

class Diffusion_model():
    def __init__(self, time_step):
        self.time_step = time_step

        self.betas = torch.tensor(np.linspace(0,1,self.time_step+2))[:-1]
        self.alphas = torch.tensor(1-self.betas)
        self.alpha_bar = torch.tensor(np.cumprod(self.alphas))

    def forward_process(self, x_0, t, noise=None):
        batch_size, n_point, _ = x_0.shape
        if noise == None:
            noise = torch.randn_like(x_0)

        a = torch.sqrt(self.alpha_bar[t])
        b = (1-self.alpha_bar[t])
        return a[:, None, None]*x_0+b[:, None, None]*noise

    def forward_one(self, x, t, noise=None):
        batch_size, n_point, _ = x.shape
        if noise == None:
            noise = torch.randn_like(x_0)

        a = torch.sqrt(1-self.betas[t])
        b = self.betas[t]
        return a[:, None, None]*x+b[:, None, None]*noise