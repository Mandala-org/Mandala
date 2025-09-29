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
import pytorch_lightning as pl
from torch import nn
from torch_scatter import scatter_add

from collections import OrderedDict

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.snapshot import Snapshot
from data.block_matrix import IrrepsBlockData
from data.graph_features import compute_graph_features

from net.common import Config, build_hidden_irreps
from net.encoders import NodeEncoder, EdgeEncoder
from net.layers import MessageBlock
from net.heads import DeepHead


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

        if cfg.train_on_energy and cfg.loss_coef_energy == 0.0:
            raise ValueError("If training on energy, loss_coef_energy must be nonzero.")
        if cfg.train_on_num_electrons and cfg.loss_coef_num_electrons == 0.0:
            raise ValueError(
                "If training on number of electrons, loss_coef_num_electrons must be nonzero."
            )
        if cfg.train_on_forces and cfg.loss_coef_forces == 0.0:
            raise ValueError("If training on forces, loss_coef_forces must be nonzero.")
        if cfg.train_on_stress and cfg.loss_coef_stress == 0.0:
            raise ValueError("If training on stress, loss_coef_stress must be nonzero.")

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = build_hidden_irreps(
            self.cfg.l_max_gnn, self.cfg.hidden_base_dim
        )
        self.neck_irreps: Irreps = build_hidden_irreps(
            self.cfg.l_max_matrix, self.cfg.hidden_base_dim
        )
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max_gnn)

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

        # ---------- message-passing stacks -----------------------------
        def _make_mp(info):
            return MessageBlock(
                self.hidden_irreps,
                self.cfg,
                initial=(
                    self.node_enc.irreps_out
                    if info["layer"] == 0 and info["graph"] == "small"
                    else None
                ),
                info=info,
            )

        self.mp_small = nn.ModuleList(
            [
                _make_mp({"layer": i, "graph": "small"})
                for i in range(self.cfg.num_layers_gnn)
            ]
        )
        self.mp_large = nn.ModuleList(
            [
                _make_mp({"layer": i, "graph": "large"})
                for i in range(self.cfg.num_layers_matrix)
            ]
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

    # ------------------------ util helpers -----------------------------
    @staticmethod
    def _mse(pred, target):
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch in block loss")
        return torch.mean((pred - target) ** 2)

    def _wrap_head_output(
        self,
        raw: Dict[str, Dict[str, torch.Tensor]],
        atoms: Tuple[str, ...],
    ) -> IrrepsBlockData:
        """
        Convert DeepHead raw dict → IrrepsBlockData with mapper.
        """
        from collections import Counter

        if self.cfg.pbc_aggregation == "sum":
            aggregated_raw = {}
            num_atoms = len(atoms)
            for key, payload in raw.items():
                vectors = payload["vectors"]
                edges = payload["edges"]

                # Create a unique ID for each (i, j) pair
                pair_ids = edges[0] * num_atoms + edges[1]

                # Find unique pairs and their inverse mapping
                unique_pair_ids, inverse_map = torch.unique(
                    pair_ids, return_inverse=True
                )

                # Sum vectors for the same pair
                summed_vectors = scatter_add(vectors, inverse_map, dim=0)

                # Get the edges for the summed vectors
                new_edges_i = unique_pair_ids // num_atoms
                new_edges_j = unique_pair_ids % num_atoms
                new_edges = torch.stack([new_edges_i, new_edges_j])

                aggregated_raw[key] = {"vectors": summed_vectors, "edges": new_edges}
            raw = aggregated_raw

        pair_vec, pair_edges, lookup = {}, {}, {}
        for key, payload in raw.items():
            vec = payload["vectors"]
            edges = payload["edges"]
            pair_vec[key] = vec
            pair_edges[key] = edges
            for idx, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, idx)

        irreps_blocks = IrrepsBlockData(
            atoms=atoms,
            atom_counts=Counter(atoms),
            pair_vectors=pair_vec,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=self.mapper.orbital_cfg,
        )
        if self.cfg.symmetrize_output:
            irreps_blocks = (irreps_blocks + irreps_blocks.transpose()) * 0.5
        return irreps_blocks

    # ------------------------------------------------------------------ forward
    def forward(self, x: Dict[str, Any]):
        # initialize activation magnitudes storage
        self._activation_mags: dict[str, torch.Tensor] = OrderedDict()

        # If edge features are not precomputed, compute them on the fly
        if not self.cfg.precompute_edge_features:
            (
                edge_index,
                edge_type_idx,
                edge_length_emb,
                edge_sh,
                index_gnn_cutoff,
                num_self_edges,
                is_closest_edge,
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
            x["edge_type_idx"] = edge_type_idx
            x["edge_length_emb"] = edge_length_emb
            x["edge_sh"] = edge_sh
            x["index_gnn_cutoff"] = index_gnn_cutoff
            x["num_self_edges"] = num_self_edges
            x["is_closest_edge"] = is_closest_edge

        # ---- encode ----------------------------------------------------
        node = self.node_enc(x["node_type_idx"], activation_mags=self._activation_mags)
        edge = self.edge_enc(
            x["edge_type_idx"],
            x["edge_length_emb"],
            x["edge_sh"],
            activation_mags=self._activation_mags,
        )
        if self.cfg.pedantic:
            assert edge.requires_grad, "Gradients not flowing through edge encoder!"

        # ---- message-passing -------------------------------------------
        index_gnn_cutoff = x["index_gnn_cutoff"]
        num_self_edges = x["num_self_edges"]

        edge_small = edge[num_self_edges:index_gnn_cutoff]
        ei_small = x["edge_index"][:, num_self_edges:index_gnn_cutoff]

        for idx, blk in enumerate(self.mp_small):
            node, edge_small = blk(
                node, edge_small, ei_small, activation_mags=self._activation_mags
            )

        edge_only_large = edge[index_gnn_cutoff:]
        edge_large = torch.cat([edge_small, edge_only_large], dim=0)
        ei_large = x["edge_index"][:, num_self_edges:]

        for idx, blk in enumerate(self.mp_large):
            node, edge_large = blk(
                node, edge_large, ei_large, activation_mags=self._activation_mags
            )

        # ---- heads -----------------------------------------------------
        # The head operates on a concatenation of node features (for self-edges)
        # and edge features (for off-diagonal edges).

        head_edge_index = x["edge_index"]
        head_edge_type_idx = x["edge_type_idx"]

        if self.cfg.pbc_aggregation == "closest":
            closest_mask = x["is_closest_edge"]
            is_closest_offdiag_mask = closest_mask[num_self_edges:]
            closest_edge_large = edge_large[is_closest_offdiag_mask]
            head_embeddings = torch.cat([node, closest_edge_large], dim=0)

            head_edge_index = x["edge_index"][:, closest_mask]
            head_edge_type_idx = x["edge_type_idx"][closest_mask]
        else:  # sum
            head_embeddings = torch.cat([node, edge_large], dim=0)

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
        # --- forward timing (message-passing + heads) -------------------
        t_fwd_start = time.perf_counter()
        preds_irreps = self(x)
        t_fwd_end = time.perf_counter()
        if self.cfg.pedantic:
            for name in self.cfg.matrix_targets:
                pred_edges = preds_irreps[name].pair_edges
                target_edges = y[name].pair_edges
                for key in target_edges.keys():
                    assert torch.equal(
                        pred_edges[key], target_edges[key]
                    ), f"Pedantic check failed: Edge order mismatch in '{name}' matrix for key '{key}'"

        # --- block mapping timing ---------------------------------------
        t_map_start = time.perf_counter()
        preds_matrix = {}
        if (
            self.cfg.train_target == "matrix"
            or self.cfg.train_on_energy
            or self.cfg.train_on_num_electrons
        ):
            for target in self.cfg.matrix_targets:
                preds_matrix[target] = preds_irreps[target].to_blocks(self.mapper)
                # symmetrize
                # preds_matrix[target] = (
                #     preds_matrix[target] + preds_matrix[target].transpose()
                # ) * 0.5
        t_map_end = time.perf_counter()

        loss_matrix = torch.tensor(0.0, device=self.cfg.device, dtype=torch.float32)
        mae_matrix = torch.tensor(0.0, device=self.cfg.device, dtype=torch.float32)

        # Select appropriate prediction and target formats for loss calculation
        if self.cfg.train_target == "matrix":
            preds_for_loss = preds_matrix
        elif self.cfg.train_target == "irreps":
            preds_for_loss = preds_irreps
        else:
            raise ValueError(f"Unknown train_target: {self.cfg.train_target}")
        targets_for_loss = y

        # Calculate matrix loss
        for name in self.cfg.matrix_targets:
            p = preds_for_loss[name]
            t = targets_for_loss[name]
            if self.cfg.train_target == "matrix":
                p_items, t_items = p.pair_blocks, t.pair_blocks
            else:  # irreps
                p_items, t_items = p.pair_vectors, t.pair_vectors

            # Vectorized loss calculation
            for key in t.keys():
                if key not in p.keys():
                    continue

                t_items_key = t_items[key]
                p_items_key = p_items[key]
                t_edges = t.pair_edges[key].t().tolist()
                p_edges = p.pair_edges[key].t().tolist()

                if self.cfg.verbosity >= 2:
                    print(f"\nMatrix '{name}', Key '{key}':")
                    print(f"  Target blocks shape: {t_items_key.shape}")
                    print(f"  Predicted blocks shape: {p_items_key.shape}")

                num_target_edges = len(t_edges)
                num_pred_edges = len(p_edges)

                pred_edge_to_idx = {tuple(edge): i for i, edge in enumerate(p_edges)}

                target_indices = []
                pred_indices = []
                for i, edge in enumerate(t_edges):
                    if tuple(edge) in pred_edge_to_idx:
                        target_indices.append(i)
                        pred_indices.append(pred_edge_to_idx[tuple(edge)])

                target_indices = torch.tensor(
                    target_indices, dtype=torch.long, device=self.device
                )
                pred_indices = torch.tensor(
                    pred_indices, dtype=torch.long, device=self.device
                )

                target_blocks_to_compare = t_items_key[target_indices]
                pred_blocks_to_compare = p_items_key[pred_indices]

                num_common_edges = len(target_indices)
                if self.cfg.verbosity >= 2:
                    print(
                        f"  Shape for loss calculation: {target_blocks_to_compare.shape}"
                    )
                perc_target_used = (
                    (num_common_edges / num_target_edges) * 100
                    if num_target_edges > 0
                    else 0
                )
                perc_pred_used = (
                    (num_common_edges / num_pred_edges) * 100
                    if num_pred_edges > 0
                    else 0
                )
                if self.cfg.verbosity >= 2:
                    print(
                        f"  Edges used: {num_common_edges}/{num_target_edges} ({perc_target_used:.2f}%) of target edges."
                    )
                    print(
                        f"  Edges used: {num_common_edges}/{num_pred_edges} ({perc_pred_used:.2f}%) of predicted edges."
                    )

                if num_common_edges > 0:
                    loss_matrix += self._mse(
                        pred_blocks_to_compare, target_blocks_to_compare
                    )
                    mae_matrix += torch.mean(
                        torch.abs(pred_blocks_to_compare - target_blocks_to_compare)
                    )

        # --- observable evaluation timing --------------------------------
        loss_E, loss_N, abs_err_E, abs_err_N = None, None, None, None
        loss_E_weighted, loss_N_weighted = torch.tensor(0.0), torch.tensor(0.0)
        t_obs_start = time.perf_counter()
        if self.cfg.enable_energy:
            if "hamiltonian" in preds_matrix and "density" in preds_matrix:
                E_pred = trace_matmul_sparse_snap_vectorized(
                    preds_matrix["hamiltonian"], preds_matrix["density"]
                )
                E_true = y["energy"]
                abs_err_E = torch.mean(torch.abs(E_pred - E_true))
                if self.cfg.train_on_energy:
                    loss_E = torch.mean((E_pred - E_true) ** 2)
                    loss_E_weighted = self.cfg.loss_coef_energy * loss_E
        if self.cfg.enable_num_electrons:
            if "overlap" in preds_matrix and "density" in preds_matrix:
                N_pred = trace_matmul_sparse_snap_vectorized(
                    preds_matrix["overlap"], preds_matrix["density"]
                )
                # electron count loss and absolute error
                N_true = y["num_electrons"]
                abs_err_N = torch.mean(torch.abs(N_pred - N_true))
                if self.cfg.train_on_num_electrons:
                    loss_N = torch.mean((N_pred - N_true) ** 2)
                    loss_N_weighted = self.cfg.loss_coef_num_electrons * loss_N
        t_obs_end = time.perf_counter()

        # total loss
        loss = loss_matrix + loss_E_weighted + loss_N_weighted

        # L1 and L2 regularization
        if self.cfg.l1_reg_coef > 0:
            l1_reg = sum(p.abs().sum() for p in self.parameters())
            loss += self.cfg.l1_reg_coef * l1_reg
        if self.cfg.l2_reg_coef > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            loss += self.cfg.l2_reg_coef * l2_reg

        # Graceful handling of NaN/inf loss
        if not torch.isfinite(loss):
            max_val = torch.finfo(loss.dtype).max
            # Log worst-case values for hyperparameter optimizer
            metrics = {
                f"{stage}_loss": torch.tensor(max_val, device=self.device),
                f"{stage}_loss_matrix": torch.tensor(max_val, device=self.device),
                # f"{stage}_loss_E": torch.tensor(max_val, device=self.device),
                # f"{stage}_loss_N": torch.tensor(max_val, device=self.device),
            }
            if loss_E is not None:
                metrics[f"{stage}_loss_E"] = torch.tensor(max_val, device=self.device)
            if loss_N is not None:
                metrics[f"{stage}_loss_N"] = torch.tensor(max_val, device=self.device)

            self.log_dict(
                metrics,
                prog_bar=True,
                on_step=self.cfg.log_on_step,
                on_epoch=self.cfg.log_on_epoch,
            )
            # Return the original non-finite loss to trigger TerminateOnNaN
            return loss

        # log all metrics
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_loss_matrix": loss_matrix,
            f"{stage}_mae_matrix": mae_matrix,
            # f"{stage}_loss_E": loss_E,
            # f"{stage}_loss_N": loss_N,
            # f"{stage}_abs_error_E": abs_err_E,
            # f"{stage}_abs_error_N": abs_err_N,
        }
        if loss_E_weighted is not None:
            metrics[f"{stage}_loss_E"] = loss_E_weighted
        if abs_err_E is not None:
            metrics[f"{stage}_abs_error_E"] = abs_err_E
        if loss_N_weighted is not None:
            metrics[f"{stage}_loss_N"] = loss_N_weighted
        if abs_err_N is not None:
            metrics[f"{stage}_abs_error_N"] = abs_err_N
        # Log percentage contributions if total loss is not zero
        if loss > 1e-8:
            metrics[f"{stage}_percent_matrix"] = (loss_matrix / loss) * 100
            if loss_E_weighted is not None:
                metrics[f"{stage}_percent_E"] = (loss_E_weighted / loss) * 100
            if loss_N_weighted is not None:
                metrics[f"{stage}_percent_N"] = (loss_N_weighted / loss) * 100

        if self.cfg.l1_reg_coef > 0:
            metrics[f"{stage}_l1_reg"] = l1_reg
        if self.cfg.l2_reg_coef > 0:
            metrics[f"{stage}_l2_reg"] = l2_reg
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

    # ------------------------------------------------------------------ optimiser
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        if not self.cfg.use_lr_scheduler:
            return optimizer

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.cfg.lr_scheduler_factor,
                patience=self.cfg.lr_scheduler_patience,
                min_lr=self.cfg.lr_scheduler_min_lr,
            ),
            "monitor": self.cfg.scheduler_target,  # Monitor the matrix loss
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
