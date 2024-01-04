import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import pytorch_lightning as pl
from sampling import Diffusion_process

class MLP(pl.LightningModule):
    def __init__(self, time_step, input_dim, hidden_dim):
        super().__init__()
        self.time_step = time_step
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.diffusion = Diffusion_process(self.time_step)

        self.time_emb = nn.Embedding(self.time_step, 2*self.input_dim)

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim*2)
        )

    def forward(self, x, t):
        t_ = self.time_emb(t)
        x_ = (x.view(-1, 2*self.input_dim)+t_).float()
        return self.layer(x_).view(-1, self.input_dim, 2)

    def generate(self):
        x_t = self.diffusion.make_noise(batch.shape)
        for t in reversed(range(self.time_step)):
            eps = self(x, t)
            x_t = self.diffusion.backward_step(x_t, t, eps)
        return x_t

    def training_step(self, batch, batch_idx):
        t = torch.randint(1, self.time_step, [batch.shape[0]])
        eps = self.diffusion.make_noise(batch.shape)
        x_t = self.diffusion.forward_process(batch, t, eps)
        predicted_eps = self(x_t, t)
        L = torch.sqrt(((eps-predicted_eps)**2).sum(2)).sum(-1)
        loss, _ = torch.var_mean(L)
        # loss = F.smooth_l1_loss(noise, predicted_noise)
        return loss

    def validation_step(self, batch, batch_idx):
        t = torch.randint(1, self.time_step, [batch.shape[0]])
        eps = self.diffusion.make_noise(batch.shape)
        x_t = self.diffusion.forward_process(batch, t, eps)
        predicted_eps = self(x_t, t)
        L = torch.sqrt(((eps-predicted_eps)**2).sum(2)).sum(-1)
        loss, _ = torch.var_mean(L)
        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer