"""
residual_e3gnn.py
===================

A PyTorch Lightning module for training a residual E3GNN model.
"""

from __future__ import annotations
from typing import Dict, Any

import torch
import pytorch_lightning as pl

from net.e3gnn import E3GNN
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config


class ResidualE3GNN(pl.LightningModule):
    """
    A pl.LightningModule that wraps a pretrained model and a new residual model.
    """

    def __init__(
        self,
        pretrained_model_path: str,
        pretrained_cfg: Config,
        residual_cfg: Config,
        mapper: BlockIrrepMapper,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["mapper"])

        self.residual_cfg = residual_cfg

        # Load and freeze the pretrained model
        self.pretrained_model = E3GNN.load_from_checkpoint(
            pretrained_model_path, mapper=mapper, cfg=pretrained_cfg
        )
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Instantiate the new model to be trained on residuals
        self.residual_model = E3GNN(mapper=mapper, cfg=residual_cfg)

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """The forward pass computes the prediction of the residual model."""
        return self.residual_model(x)

    def training_step(self, batch, batch_idx):
        x, y_residual_matrix, _ = batch

        # Get residual prediction
        y_pred_residual_irreps = self.forward(x)
        y_pred_residual_matrix = {
            name: p.to_blocks(self.residual_model.mapper)
            for name, p in y_pred_residual_irreps.items()
        }

        # Compute loss against the residual target
        loss_dict = self.residual_model.compute_loss(
            y_pred_residual_matrix, y_residual_matrix
        )
        loss = loss_dict["loss_total"]

        # Log training metrics
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_residual_matrix, y_true_matrix = batch

        # Get predictions from both models
        y_pred_residual_irreps = self.forward(x)
        with torch.no_grad():
            y_pred_pretrained_irreps = self.pretrained_model(x)

        # Convert to BlockMatrix
        mapper = self.residual_model.mapper
        y_pred_residual_matrix = {
            name: p.to_blocks(mapper) for name, p in y_pred_residual_irreps.items()
        }
        y_pred_pretrained_matrix = {
            name: p.to_blocks(mapper) for name, p in y_pred_pretrained_irreps.items()
        }

        # Combine predictions
        y_pred_total_matrix = {
            name: y_pred_pretrained_matrix[name] + y_pred_residual_matrix[name]
            for name in y_pred_residual_matrix
        }

        # 1. Calculate and log metrics for the combined model against the true target
        obs_metrics = self.residual_model.compute_observables(
            y_pred_total_matrix, y_true_matrix
        )
        for key, value in obs_metrics.items():
            self.log(f"val/combined_{key}", value, prog_bar=True)

        # 2. Calculate and log loss for the residual model against the residual target
        residual_loss_dict = self.residual_model.compute_loss(
            y_pred_residual_matrix, y_residual_matrix
        )
        for key, value in residual_loss_dict.items():
            self.log(f"val/residual_{key}", value, prog_bar=True)

    def configure_optimizers(self):
        """Configure the optimizer for the residual model's parameters."""
        optimizer = torch.optim.AdamW(
            self.residual_model.parameters(), lr=self.residual_cfg.lr
        )
        if not self.residual_cfg.use_lr_scheduler:
            return optimizer

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.residual_cfg.lr_scheduler_factor,
                patience=self.residual_cfg.lr_scheduler_patience,
                min_lr=self.residual_cfg.lr_scheduler_min_lr,
            ),
            "monitor": "val/residual_loss_matrix_total",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
