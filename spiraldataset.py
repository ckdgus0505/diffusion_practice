import os
import numpy as np
import torch
from torch.utils.data import Dataset

def spiral(N=400):
  pi=np.pi
  theta = np.sqrt(np.random.rand(N))*4*pi
  r_a = 2*theta + pi
  data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T

  x_a = data_a + np.clip(np.random.randn(N,2), -2,2)
  np.random.shuffle(x_a)
  return x_a/(3*pi+2)

class SpiralDataset(Dataset):
  def __init__(self, n_dataset, n_sample, save_data=False, data_path=None):
    self.n_dataset = n_dataset
    self.n_sample = n_sample
    self.save_data=False
    self.data_path=data_path

    if self.data_path is not None and save_data:
      os.makedirs(self.data_path, exist_ok=True)
      for i in range(self.n_dataset):
        np.save(os.path.join(self.data_path, str(i)+'.npy'), spiral())

  def __getitem__(self, idx):
    if self.data_path is not None and save_data:
      return np.load(os.path.join(self.dir, str(idx)+'.npy'))
    else:
      return spiral(self.n_sample)

  def __len__(self):
    return self.n_dataset