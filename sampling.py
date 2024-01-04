import numpy as np
import torch

class Diffusion_process():
    def __init__(self, time_step):
        self.time_step = time_step

        self.betas = torch.tensor(np.linspace(0,1,self.time_step+2)[:-1]).requires_grad_(False)
        self.alphas = torch.tensor(1-self.betas).requires_grad_(False)
        self.alpha_bar = torch.tensor(np.cumprod(self.alphas)).requires_grad_(False)

    def make_noise(self, shape):
        return torch.randn(shape)

    def forward_process(self, x_0, t, noise=None):
        batch_size, n_point, _ = x_0.shape
        if noise == None:
            noise = torch.randn_like(x_0)

        a = torch.sqrt(self.alpha_bar[t])
        b = (1-self.alpha_bar[t])
        return a[:, None, None]*x_0+b[:, None, None]*noise

    def forward_step(self, x, t, noise=None):
        batch_size, n_point, _ = x.shape
        if noise == None:
            noise = torch.randn_like(x)

        a = torch.sqrt(1-self.betas[t])
        b = self.betas[t]
        return a[:, None, None]*x+b[:, None, None]*noise

    def backward_step(self, x_t, t, eps):
        z = torch.zeros_like(x_t)
        if t > 1:
            z = torch.randn_like(x_t)

        a = (1/torch.sqrt(self.alphas[t]))
        b = (self.betas[t]/torch.sqrt(1-self.alpha_bar[t]))
        c = (1-self.alpha_bar[t-1])/(1-self.alpha_bar[t])*(self.betas[t])
        return a[:, None, None]*(x_t-b[:, None, None]*eps)+c[:, None, None]*z