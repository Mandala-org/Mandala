"""
residual_dataset.py
===================

Dataset that wraps an E3GNNDataset and a pretrained model to yield residuals
(y_true - y_pred) as the new training targets.
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from data.gnn_dataset import E3GNNDataset
from net.e3gnn import E3GNN
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config


class ResidualDataset(Dataset):
    """
    Wraps an E3GNNDataset and a pretrained model to create a dataset of residuals.

    For each sample (x, y_true) in the original dataset, this dataset yields
    (x, y_residual, y_true), where y_residual = y_true - y_pred.
    """

    def __init__(
        self,
        original_dataset: E3GNNDataset,
        pretrained_model_path: str,
        pretrained_cfg: Config,
        mapper: BlockIrrepMapper,
        device: str | torch.device = "cpu",
    ):
        self.original_dataset = original_dataset
        self.device = device

        # Load the pretrained model
        print(f"Loading pretrained model from {pretrained_model_path}...")
        self.pretrained_model = E3GNN.load_from_checkpoint(
            pretrained_model_path, mapper=mapper, cfg=pretrained_cfg
        )
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Pre-compute residuals for all samples
        self.samples: List[Tuple[Dict, Dict, Dict]] = []
        print("Calculating residuals...")
        with torch.no_grad():
            for i in tqdm(
                range(len(self.original_dataset)), desc="Generating Residuals"
            ):
                x, y_true = self.original_dataset[i]

                # Move data to the correct device for the model
                x_device = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()
                }

                # Get prediction from the pretrained model
                y_pred_irreps = self.pretrained_model(x_device)

                # Convert predictions and ground truth to BlockMatrix on CPU
                y_pred_matrix = {
                    name: p.to_blocks(mapper).to("cpu")
                    for name, p in y_pred_irreps.items()
                }
                y_true_matrix = {name: m.to("cpu") for name, m in y_true.items()}

                # Calculate residual: y_residual = y_true - y_pred
                y_residual_matrix = {}
                for name in y_true_matrix:
                    if name in y_pred_matrix:
                        y_residual_matrix[name] = (
                            y_true_matrix[name] - y_pred_matrix[name]
                        )

                # Store the sample with original input, residual target, and true target
                self.samples.append((x, y_residual_matrix, y_true_matrix))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict]:
        """
        Returns a tuple of (x, y_residual_matrix, y_true_matrix).
        """
        return self.samples[idx]
