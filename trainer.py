import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TrackStateMLP(pl.LightningModule):

    def __init__(self, input_dim=84, state_dim=5, hidden_dim=64, num_layers=3, lr=1e-3, dropout=0.1, phi_mean=0, phi_std=1):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.state_dim = state_dim
        self.phi_mean = phi_mean
        self.phi_std = phi_std

        # Backbone
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Track-state prediction
        self.state_head = nn.Linear(hidden_dim, state_dim)

        # Loss
        self.criterion = nn.MSELoss()

    """
        Parameters
        ----------
        x_cont : torch.Tensor
        Shape: [batch_size, 39]
        
        Returns
        -------
        state : torch.Tensor
        Shape: [batch_size, 5]
        
        [d0, phi0, kappa, z0, tandip]
    """
    def forward(self, x_cont):
        h = self.backbone(x_cont)
        state = self.state_head(h)
        return state

    def _compute_loss(self, pred_state, target_state):
        diff = pred_state - target_state

        phi_mean = self.phi_mean
        phi_std = self.phi_std

        pred_phi = pred_state[:, 1] * phi_std + phi_mean
        true_phi = target_state[:, 1] * phi_std + phi_mean

        phi_diff = torch.atan2(
            torch.sin(pred_phi - true_phi),
            torch.cos(pred_phi - true_phi)
        )

        phi_diff = phi_diff / phi_std

        diff = diff.clone()
        diff[:, 1] = phi_diff

        loss = torch.mean(diff ** 2)
        #loss = self.criterion(pred_state, target_state)
        return loss

    def training_step(self, batch, batch_idx):
        x_cont, target_state = batch
        pred_state = self(x_cont)
        loss = self._compute_loss(pred_state, target_state)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_cont, target_state = batch
        pred_state = self(x_cont)
        loss = self._compute_loss(pred_state, target_state)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class LossTracker(Callback):
    """
    PyTorch Lightning callback to track
    training and validation losses.
    """

    def __init__(self):
        super().__init__()

        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())