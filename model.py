import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sampling import Diffusion_process

class MLP(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.time_step = cfg["model"]["time_step"]
        self.input_dim = cfg["model"]["input_dim"]
        self.hidden_dim = cfg["model"]["hidden_dim"]
        self.diffusion = Diffusion_process(self.time_step)

        self.time_emb = nn.Embedding(self.time_step, 2*self.input_dim)

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim*2)
        )

        self.training_step_outputs = 0
        self.val_step_outputs = 0
        self.n_training_batch=0
        self.n_val_batch=0

    def forward(self, x, t):
        t_ = self.time_emb(torch.tensor(t).to(self.device))
        x_ = (x.view(-1, 2*self.input_dim)+t_).float()
        return self.layer(x_).view(-1, self.input_dim, 2)

    def generate(self, N=1):
        x_t = self.diffusion.make_noise([N, self.input_dim, 2]).to(self.device)
        for t in reversed(range(self.time_step)):
            t = torch.tensor(t).to(self.device)
            eps = self(x_t, t)
            x_t = self.diffusion.backward_step(x_t, t, eps)
        return x_t

    def training_step(self, batch, batch_idx):
        t = torch.randint(1, self.time_step+1, [batch.shape[0]]).to(self.device)
        eps = self.diffusion.make_noise(batch.shape).to(self.device)
        x_t = self.diffusion.forward_process(batch, t, eps)
        predicted_eps = self(x_t, t)
        loss = F.smooth_l1_loss(eps, predicted_eps)
        self.training_step_outputs+=loss.item()
        self.n_training_batch = max(self.n_training_batch, batch_idx+1)
        return loss
    
    def on_train_epoch_end(self):
        self.log("training_loss_epoch", self.training_step_outputs/self.n_training_batch, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs=0

    def validation_step(self, batch, batch_idx):
        t = torch.randint(1, self.time_step+1, [batch.shape[0]]).to(self.device)
        eps = self.diffusion.make_noise(batch.shape).to(self.device)
        x_t = self.diffusion.forward_process(batch, t, eps)
        predicted_eps = self(x_t, t)
        loss = F.smooth_l1_loss(eps, predicted_eps)
        self.val_step_outputs+=loss.item()
        self.n_val_batch = max(self.n_val_batch, batch_idx+1)
        return loss

    def on_validation_epoch_end(self):
        self.log("validation_loss_epoch", self.val_step_outputs/self.n_val_batch, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs=0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer