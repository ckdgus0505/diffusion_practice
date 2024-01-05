import yaml
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import MLP
from spiraldataset import SpiralDataset

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main():
    with open('config.yaml') as file:
        cfg = yaml.safe_load(file)

    model = MLP(cfg)

    train_set = SpiralDataset(cfg["dataset"]["train_ds"]["n_dataset"], cfg["dataset"]["n_sample"], save_data=False, data_path=None)
    valid_set = SpiralDataset(cfg["dataset"]["valid_ds"]["n_dataset"], cfg["dataset"]["n_sample"], save_data=True, data_path="valid")
    train_loader = DataLoader(train_set, batch_size=batch, drop_last=True, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_set, batch_size=batch, drop_last=True, shuffle=True, num_workers=1)

    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    main()