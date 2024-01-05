import numpy as np
import torch

class Diffusion_process():
    def __init__(self, time_step):
        super().__init__()
        self.time_step = time_step

        self.betas = np.append(0, np.linspace(1e-4, 0.02, self.time_step))
        self.alphas = 1-self.betas
        self.alpha_bar = np.cumprod(self.alphas)

    def make_noise(self, shape):
        return torch.randn(shape)

    def forward_process(self, x_0, t, noise=None):
        alpha_bar = torch.tensor(self.alpha_bar, device=x_0.device)
        batch_size, n_point, _ = x_0.shape
        if noise == None:
            noise = torch.randn_like(x_0, device = x_0.device)

        a = torch.sqrt(alpha_bar[t])
        b = torch.sqrt(1-alpha_bar[t])
        return a[:, None, None]*x_0+b[:, None, None]*noise

    def forward_step(self, x, t, noise=None):
        betas = torch.tensor(self.betas, device=x.device)
        batch_size, n_point, _ = x.shape
        if noise == None:
            noise = torch.randn_like(x, device = x.device)

        a = torch.sqrt(1-betas[t])
        b = betas[t]
        return a[:, None, None]*x+b[:, None, None]*noise

    def backward_step(self, x_t, t, eps):
        betas = torch.tensor(self.betas, device=x_t.device)
        alphas = torch.tensor(self.alphas, device=x_t.device)
        alpha_bar = torch.tensor(self.alpha_bar, device=x_t.device)
        if t == 1:
            z = torch.zeros_like(x_t).to(x_t.device)
        else :
            z = torch.randn_like(x_t).to(x_t.device)
        
        a = (1/torch.sqrt(alphas[t]))
        b = (1-alphas[t]/torch.sqrt(1-alpha_bar[t]))
        c = (1-alpha_bar[t-1])/(1-alpha_bar[t])*(betas[t])
        return a[:, None, None]*(x_t-b[:, None, None]*eps)+c[:, None, None]*z