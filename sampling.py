import torch

def forward_process(x_0, t, noise=None):
  batch_size, n_point, _ = x_0.shape
  if noise == None:
    noise = torch.randn_like(x_0)

  a = torch.sqrt(alpha_bar[t])
  b = (1-alpha_bar[t])
  return a[:, None, None]*x_0+b[:, None, None]*noise

def forward_one(x, t, noise=None):
  batch_size, n_point, _ = x.shape
  if noise == None:
    noise = torch.randn_like(x_0)

  a = torch.sqrt(1-betas[t])
  b = betas[t]
  return a[:, None, None]*x+b[:, None, None]*noise