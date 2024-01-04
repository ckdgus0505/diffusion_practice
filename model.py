import torch
import torch.nn
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    def __init__(self, time_step, input_dim, hidden_dim):
        super().__init__()
        self.time_step = time_step
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.time_emb = nn.Embedding(self.time_step, 2*input_dim)

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim*2)
        )

    def forward(self, x, t):
        t_ = self.time_emb(t)
        x_ = x.view(-1, 2*input_dim)+t_
        return self.layer(x_).view(-1, self.input_dim, 2)