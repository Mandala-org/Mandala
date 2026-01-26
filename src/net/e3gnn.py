"""
E3GNN – PyTorch-Lightning implementation
----------------------------------------

*  encoders.py         → NodeEncoder / EdgeEncoder
*  layers.py           → MessageBlock (small & large graphs)
*  heads.py            → DeepHead (H, S, D)
*  sparse_math.trace_* → Energy / electron count losses
"""

from __future__ import annotations
from typing import Dict, Tuple, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn

from collections import OrderedDict

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_block_matrix
from data.snapshot import Snapshot
from data.block_matrix import IrrepsBlockData
from data.graph_features import compute_graph_features

from net.common import Config, build_hidden_irreps
from net.encoders import NodeEncoder, EdgeEncoder
from net.layers import MessageBlock
from net.heads import DeepHead
from utils.summary import print_model_summary
from utils.units import HARTREE_TO_EV


class E3GNN(pl.LightningModule):
    """
    Full network **and** training logic.

    Forward returns a dict with three `IrrepsBlockData` entries
    (hamiltonian / overlap / density).  Losses are computed inside
    `training_step`.
    """

    def __init__(
        self,
        mapper: BlockIrrepMapper,
        cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        # The mapper is now an nn.Module and will be moved to the correct device
        # automatically by PyTorch Lightning.
        self.mapper: BlockIrrepMapper = mapper

        if cfg.train_on_energy and cfg.loss_coef_observables == 0.0:
            raise ValueError(
                "If training on energy, loss_coef_observables must be nonzero."
            )
        if cfg.train_on_num_electrons and cfg.loss_coef_observables == 0.0:
            raise ValueError(
                "If training on number of electrons, loss_coef_observables must be nonzero."
            )
        if cfg.train_on_forces and cfg.loss_coef_forces == 0.0:
            raise ValueError("If training on forces, loss_coef_forces must be nonzero.")
        if cfg.train_on_stress and cfg.loss_coef_stress == 0.0:
            raise ValueError("If training on stress, loss_coef_stress must be nonzero.")

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = build_hidden_irreps(
            self.cfg.l_max, self.cfg.hidden_base_dim, self.cfg.emb_use_odd_features
        )
        self.neck_irreps: Irreps = build_hidden_irreps(
            self.cfg.l_max,
            self.cfg.hidden_base_dim,
            self.cfg.emb_use_odd_features,
        )
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max)

        # ---------- encoders -------------------------------------------
        self.node_enc = NodeEncoder(
            node_one_hot_dim=len(self.mapper.orbital_cfg.elements()),
            cfg=self.cfg,
            info={"name": "node_encoding"},
        )
        self.edge_enc = EdgeEncoder(
            n_edge_types=len(self.mapper._maps.keys()),
            irreps_out=self.hidden_irreps,
            cfg=self.cfg,
            info={"name": "edge_encoding"},
        )

        # ---------- message-passing ------------------------------
        # First layer: nodes start as scalars from node_enc
        # Subsequent layers: nodes have full hidden_irreps after first NodeUpdateBlock expansion
        self.mp_blocks = nn.ModuleList()
        for i in range(self.cfg.num_layers_gnn):
            # First layer uses node_enc output (scalars), rest use hidden_irreps
            node_irreps_in = self.node_enc.irreps_out if i == 0 else self.hidden_irreps
            self.mp_blocks.append(
                MessageBlock(
                    node_irreps=node_irreps_in,
                    edge_irreps=self.hidden_irreps,
                    num_species=len(self.mapper.orbital_cfg.elements()),
                    cfg=self.cfg,
                    info={"layer": i},
                )
            )

        # ---------- heads ----------------------------------------------
        pair_keys = list(
            map(lambda pair: f"{pair[0]}-{pair[1]}", self.mapper._maps.keys())
        )
        self.heads = nn.ModuleDict(
            {
                name: DeepHead(
                    irreps_hidden=self.hidden_irreps,
                    irreps_neck=self.neck_irreps,
                    pair_keys=pair_keys,
                    mapper=self.mapper,
                    cfg=self.cfg,
                    info={"matrix": name},
                )
                for name in self.cfg.matrix_targets
            }
        )

        # Print model summary if verbosity >= 1
        print_model_summary(self, verbosity=self.cfg.verbosity)

    # ------------------------ util helpers -----------------------------
    @staticmethod
    def _mse(pred, target):
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch in block loss: {pred.shape=}, {target.shape=}"
            )
        return torch.mean((pred - target) ** 2)

    @staticmethod
    def _mae(pred, target):
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch in block loss: {pred.shape=}, {target.shape=}"
            )
        return torch.mean(torch.abs(pred - target))

    def _wrap_head_output(
        self,
        raw: Dict[str, Dict[str, torch.Tensor]],
        atoms: Tuple[str, ...],
    ) -> IrrepsBlockData:
        """
        Convert DeepHead raw dict → IrrepsBlockData with mapper.
        """
        from collections import Counter

        pair_vec, pair_edges, lookup = {}, {}, {}
        for key, payload in raw.items():
            vec = payload["vectors"]
            edges = payload["edges"]
            pair_vec[key] = vec
            pair_edges[key] = edges
            for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist()):
                lookup[(sx, sy, sz, i, j)] = (key, idx)

        irreps_blocks = IrrepsBlockData(
            atoms=atoms,
            atom_counts=Counter(atoms),
            pair_vectors=pair_vec,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=self.mapper.orbital_cfg,
        )
        return irreps_blocks

    # ------------------------------------------------------------------ forward
    def forward(self, x: Dict[str, Any]):
        # initialize activation magnitudes storage
        self._activation_mags: dict[str, torch.Tensor] = OrderedDict()

        # If edge features are not precomputed, compute them on the fly
        if not self.cfg.precompute_edge_features:
            (
                edge_index,
                edge_shift,
                edge_type_idx,
                edge_length_emb,
                edge_sh,
                index_gnn_cutoff,
                num_self_edges,
            ) = compute_graph_features(
                positions=x["positions"],
                box=x["box"],
                atoms=x["atoms"],
                cfg=self.cfg,
                sh_irreps=self.sh_irreps,
                edge_type2idx=self.mapper.edge_type2idx,
            )

            # Update x with the newly computed features
            x["edge_index"] = edge_index
            x["edge_shift"] = edge_shift
            x["edge_type_idx"] = edge_type_idx
            x["edge_length_emb"] = edge_length_emb
            x["edge_sh"] = edge_sh
            x["index_gnn_cutoff"] = index_gnn_cutoff
            x["num_self_edges"] = num_self_edges

        # ---- encode ----------------------------------------------------
        node = self.node_enc(x["node_type_idx"], activation_mags=self._activation_mags)
        edge = self.edge_enc(
            x["edge_type_idx"],
            x["edge_length_emb"],
            x["edge_sh"],
            activation_mags=self._activation_mags,
        )
        if self.cfg.safety_checks and torch.is_grad_enabled():
            assert node.requires_grad, "Gradients not flowing through node encoder!"
            assert edge.requires_grad, "Gradients not flowing through edge encoder!"

        # ---- message-passing -------------------------------------------
        # Prepare one-hot encodings for self-connections
        num_species = len(self.mapper.orbital_cfg.elements())
        node_one_hot = F.one_hot(x["node_type_idx"], num_classes=num_species).float()

        # Edge one-hot: encode pairs of node types
        src_type = x["node_type_idx"][x["edge_index"][0]]
        dst_type = x["node_type_idx"][x["edge_index"][1]]
        edge_one_hot = F.one_hot(
            src_type * num_species + dst_type, num_classes=num_species * num_species
        ).float()

        # Single unified message-passing loop
        for idx, blk in enumerate(self.mp_blocks):
            node, edge = blk(
                node=node,
                edge=edge,
                edge_index=x["edge_index"],
                edge_sh=x["edge_sh"],
                edge_length_emb=x["edge_length_emb"],
                node_one_hot=node_one_hot,
                edge_one_hot=edge_one_hot,
                activation_mags=self._activation_mags,
            )

        # ---- heads -----------------------------------------------------
        # The head operates on a concatenation of node features (for self-edges)
        # and edge features (for off-diagonal edges).

        # concatenate edge_shift with edge_index
        head_edge_index = torch.cat([x["edge_shift"], x["edge_index"]], dim=0)
        head_edge_type_idx = x["edge_type_idx"]

        if self.cfg.head_use_self_edges:
            head_embeddings = edge
        else:
            head_embeddings = torch.cat([node, edge[x["num_self_edges"] :]], dim=0)

        preds_raw = {
            name: head(head_embeddings, head_edge_type_idx, head_edge_index)
            for name, head in self.heads.items()
        }

        preds_wrapped = {
            name: self._wrap_head_output(raw, tuple(x["atoms"]))
            for name, raw in preds_raw.items()
        }

        # expose activation magnitudes for callbacks
        self._last_activation_mags = self._activation_mags
        return preds_wrapped

    # ==================== Lightning steps ====================================
    def _shared_step(self, batch, batch_idx, stage: str):
        """
        Shared logic for training and validation steps.
        Logs metrics prefixed with stage ('train' or 'val').
        """
        x, y = batch
        metrics = {}

        # --- forward timing (message-passing + heads) -------------------
        t_fwd_start = time.perf_counter()
        preds_irreps = self(x)
        t_fwd_end = time.perf_counter()

        # --- block mapping timing ---------------------------------------
        t_map_start = time.perf_counter()
        preds_matrix = {
            name: preds_irreps[name].to_blocks(self.mapper)
            for name in self.cfg.matrix_targets
        }
        t_map_end = time.perf_counter()

        # --- Matrix Loss Calculation ------------------------------------
        matrix_mses = {}
        matrix_maes = {}
        combined_matrix_losses = {}
        preds_for_loss = (
            preds_irreps if self.cfg.train_target == "irreps" else preds_matrix
        )

        # num_atoms = x["positions"].shape[0]

        for name in self.cfg.matrix_targets:
            p = preds_for_loss[name]
            t = y[name]

            if self.cfg.train_target == "matrix" and self.cfg.symmetrize_output:
                p = (p + p.transpose()) * 0.5

            p_items, t_items = (
                (p.pair_vectors, t.pair_vectors)
                if self.cfg.train_target == "irreps"
                else (p.pair_blocks, t.pair_blocks)
            )

            mse_val = torch.tensor(0.0, device=self.device)
            mae_val = torch.tensor(0.0, device=self.device)

            # Vectorized loss calculation
            for key in t_items.keys():
                if key not in p_items.keys():
                    raise ValueError(f"Key {key} not found in predicted items.")
                preds = p_items[key]
                targets = t_items[key]

                # Handle size mismatch by truncating to the smaller size
                # if preds.shape[0] > targets.shape[0] that means that cutoff_radius
                # is bigger than maximum distance in the matrix
                # if preds.shape[0] < targets.shape[0] that means that the maximum
                # distance in the matrix is bigger than cutoff_radius
                min_n = min(preds.shape[0], targets.shape[0])
                preds = preds[:min_n]
                targets = targets[:min_n]

                if self.cfg.safety_checks:
                    assert torch.equal(
                        p.pair_edges[key][:, :min_n], t.pair_edges[key][:, :min_n]
                    ), f"Edge mismatch in block loss for matrix {name}, key {key}."

                mse_val += self._mse(preds, targets)
                mae_val += self._mae(preds, targets)

            matrix_mses[name] = mse_val
            matrix_maes[name] = mae_val

            # Store for combined loss BEFORE unit conversion
            mse_for_loss = mse_val
            mae_for_loss = mae_val

            if name == "hamiltonian":
                # Convert to eV^2 and eV for logging only
                mse_val = mse_val * (HARTREE_TO_EV**2)
                mae_val = mae_val * HARTREE_TO_EV
            metrics[f"{stage}/{name}_mae"] = mae_val
            metrics[f"{stage}/{name}_mse"] = mse_val

            combined_matrix_losses[name] = (
                1 - self.cfg.loss_l1_fraction
            ) * mse_for_loss + self.cfg.loss_l1_fraction * mae_for_loss

        loss_matrix = sum(combined_matrix_losses.values())

        # --- Observable Evaluation --------------------------------------
        t_obs_start = time.perf_counter()
        loss_E_weighted = torch.tensor(0.0, device=self.device)
        loss_N_weighted = torch.tensor(0.0, device=self.device)

        # Standard observables
        if (
            self.cfg.enable_energy
            and "hamiltonian" in preds_matrix
            and "density" in preds_matrix
        ):
            E_pred = trace_matmul_sparse_block_matrix(
                preds_matrix["hamiltonian"], preds_matrix["density"]
            )
            E_true = y["energy"]
            metrics[f"{stage}/energy_mae"] = (
                torch.mean(torch.abs(E_pred - E_true)) * HARTREE_TO_EV
            )
            if self.cfg.train_on_energy and not self.cfg.train_observables_on_gt:
                loss_E_weighted = self.cfg.loss_coef_observables * self._mse(
                    E_pred, E_true
                )

        if (
            self.cfg.enable_num_electrons
            and "overlap" in preds_matrix
            and "density" in preds_matrix
        ):
            N_pred = trace_matmul_sparse_block_matrix(
                preds_matrix["overlap"], preds_matrix["density"]
            )
            N_true = y["num_electrons"]
            metrics[f"{stage}/num_electrons_mae"] = torch.mean(
                torch.abs(N_pred - N_true)
            )
            if self.cfg.train_on_num_electrons and not self.cfg.train_observables_on_gt:
                loss_N_weighted = self.cfg.loss_coef_observables * self._mse(
                    N_pred, N_true
                )

        # Partial Ground Truth Observables
        if self.cfg.log_partial_gt_observables or self.cfg.train_observables_on_gt:
            if self.cfg.train_target == "irreps":
                H_true = y["hamiltonian"].to_blocks(self.mapper)
                D_true = y["density"].to_blocks(self.mapper)
                S_true = y["overlap"].to_blocks(self.mapper)
            else:
                H_true = y["hamiltonian"]
                D_true = y["density"]
                S_true = y["overlap"]

            E_gt_D = trace_matmul_sparse_block_matrix(
                preds_matrix["hamiltonian"], D_true
            )
            E_gt_H = trace_matmul_sparse_block_matrix(H_true, preds_matrix["density"])
            N_gt_S = trace_matmul_sparse_block_matrix(preds_matrix["density"], S_true)
            N_gt_D = trace_matmul_sparse_block_matrix(S_true, preds_matrix["density"])

            if self.cfg.log_partial_gt_observables:
                metrics[f"{stage}/energy_mae_gt_density"] = (
                    torch.mean(torch.abs(E_gt_D - E_true)) * HARTREE_TO_EV
                )
                metrics[f"{stage}/energy_mae_gt_hamiltonian"] = (
                    torch.mean(torch.abs(E_gt_H - E_true)) * HARTREE_TO_EV
                )
                metrics[f"{stage}/num_electrons_mae_gt_overlap"] = torch.mean(
                    torch.abs(N_gt_S - N_true)
                )
                metrics[f"{stage}/num_electrons_mae_gt_density"] = torch.mean(
                    torch.abs(N_gt_D - N_true)
                )

            if self.cfg.train_on_energy and self.cfg.train_observables_on_gt:
                loss_E_gt_D = self._mse(E_gt_D, E_true)
                loss_E_gt_H = self._mse(E_gt_H, E_true)
                loss_E_weighted = (
                    self.cfg.loss_coef_observables * (loss_E_gt_D + loss_E_gt_H) * 0.5
                )

            if self.cfg.train_on_num_electrons and self.cfg.train_observables_on_gt:
                loss_N_gt_S = self._mse(N_gt_S, N_true)
                loss_N_gt_D = self._mse(N_gt_D, N_true)
                loss_N_weighted = (
                    self.cfg.loss_coef_observables * (loss_N_gt_S + loss_N_gt_D) * 0.5
                )

        t_obs_end = time.perf_counter()

        # --- Total Loss Aggregation ---
        total_matrix_l1_component = self.cfg.loss_l1_fraction * sum(
            matrix_maes.values()
        )
        total_matrix_l2_component = (1 - self.cfg.loss_l1_fraction) * sum(
            matrix_mses.values()
        )

        total_l1_loss = total_matrix_l1_component
        total_l2_loss = total_matrix_l2_component + loss_E_weighted + loss_N_weighted

        loss = total_l1_loss + total_l2_loss

        # L1 and L2 regularization
        if self.cfg.l1_reg_coef > 0:
            l1_reg = sum(p.abs().sum() for p in self.parameters())
            loss += self.cfg.l1_reg_coef * l1_reg
            metrics[f"{stage}/loss_l1_reg"] = l1_reg
        if self.cfg.l2_reg_coef > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            loss += self.cfg.l2_reg_coef * l2_reg
            metrics[f"{stage}/loss_l2_reg"] = l2_reg

        # --- Logging ------------------------------------------------------
        metrics[f"{stage}/loss_total"] = loss
        metrics[f"{stage}/loss_matrix_total"] = loss_matrix

        if loss_E_weighted > 0:
            metrics[f"{stage}/loss_energy"] = loss_E_weighted
        if loss_N_weighted > 0:
            metrics[f"{stage}/loss_num_electrons"] = loss_N_weighted

        if stage == "train" and loss > 1e-12:
            for name, val in combined_matrix_losses.items():
                metrics[f"frac/loss_{name}"] = val / loss
            if loss_E_weighted > 0:
                metrics["frac/loss_energy"] = loss_E_weighted / loss
            if loss_N_weighted > 0:
                metrics["frac/loss_num_electrons"] = loss_N_weighted / loss

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=self.cfg.log_on_step,
            on_epoch=self.cfg.log_on_epoch,
            batch_size=1,
        )

        # record per-batch timings for callback
        self._last_batch_times = {
            "forward": t_fwd_end - t_fwd_start,
            "map": t_map_end - t_map_start,
            "obs": t_obs_end - t_obs_start,
        }
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def on_train_epoch_start(self):
        """Log learning rate at the beginning of each training epoch."""
        if self.trainer.sanity_checking:
            return
        # Get the first optimizer
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]

        lr = optimizer[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # ------------------------------------------------------------------ optimiser
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        if not self.cfg.use_lr_scheduler:
            return optimizer

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.cfg.lr_scheduler_factor,
                patience=self.cfg.lr_scheduler_patience,
                min_lr=self.cfg.lr_scheduler_min_lr,
            ),
            "monitor": self.cfg.lr_scheduler_target,  # Monitor the matrix loss
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------ force prediction
    def predictions_to_snapshot(
        self,
        predictions: Dict[str, IrrepsBlockData],
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> "Snapshot":
        return Snapshot(
            hamiltonian=(
                predictions["hamiltonian"].to_blocks(self.mapper)
                if "hamiltonian" in predictions
                else None
            ),
            overlap=(
                predictions["overlap"].to_blocks(self.mapper)
                if "overlap" in predictions
                else None
            ),
            density=(
                predictions["density"].to_blocks(self.mapper)
                if "density" in predictions
                else None
            ),
            positions=positions,
            box=box,
        )

    def get_forces(
        self,
        predictions: Dict[str, IrrepsBlockData],
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute forces from predictions and positions.
        Args:
            predictions: Output from the forward pass, containing hamiltonian,
                         overlap, and density matrices.
            positions: Atomic positions (N, 3).
            box: Lattice box matrix (3, 3).
        Returns:
            Forces as a tensor of shape (N, 3).
        Comments:
            - Forces are computed as -∂E/∂r, where E is the energy from the hamiltonian.
            - No Pulay correction needed
        """
        snapshot = self.predictions_to_snapshot(predictions, positions, box)
        energy = snapshot.get_energy()
        grad_pos = torch.autograd.grad(
            energy,
            positions,
            create_graph=self.cfg.train_on_forces,  # needed for second derivatives
            retain_graph=True,
        )[0]
        return -grad_pos

    def get_box_grad(
        self,
        predictions: Dict[str, IrrepsBlockData],
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of energy with respect to box.
        Args:
            predictions: Output from the forward pass, containing hamiltonian,
                         overlap, and density matrices.
            positions: Atomic positions (N, 3).
            box: Lattice box matrix (3, 3).
        Returns:
            Gradient of energy with respect to box as a tensor of shape (3, 3).
        """
        snapshot = self.predictions_to_snapshot(predictions, positions, box)
        energy = snapshot.get_energy()
        # 1. get dE/dh
        grad_box = torch.autograd.grad(
            energy,
            box,
            create_graph=self.cfg.train_on_stress,  # needed for second derivatives
            retain_graph=True,
        )[0]
        return grad_box

    def get_stress(
        self,
        predictions: Dict[str, IrrepsBlockData],
        positions: torch.Tensor,
        box: torch.Tensor,
        symmetrize: bool = True,
    ) -> torch.Tensor:
        """
        Compute stress tensor from energy and box.
        Args:
            energy: Scalar energy value.
            box: Lattice box matrix (3, 3).
        Returns:
            Stress tensor as a (3, 3) tensor.
        Comments:
            - Stress is defined as σ_{αβ} = (1/Ω) ∑_γ h_{γα} (dE/dh_{γβ})
            - No Pulay correction needed
        """
        # 1. get dE/dh
        grad_box = self.get_box_grad(predictions, positions, box)
        # 2. compute volume
        volume = torch.det(box)
        # 3. compute stress tensor σ_{αβ} = (1/Ω) ∑_γ h_{γα} (dE/dh_{γβ})
        # stress = (1/Ω) * box^T @ grad_box
        stress = torch.matmul(box.t(), grad_box) / volume
        # 4. optionally symmetrize: σ → (σ+σ^T)/2
        if symmetrize:
            stress = 0.5 * (stress + stress.transpose(-1, -2))
        return stress

    def predict_forces(self, x: Dict[str, Any]) -> torch.Tensor:
        predictions = self(x)
        return self.get_forces(predictions, x["positions"], x["box"])

    def predict_stress(self, x: Dict[str, Any]) -> torch.Tensor:
        predictions = self(x)
        return self.get_stress(predictions, x["positions"], x["box"])
