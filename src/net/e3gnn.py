"""
E3GNN – PyTorch-Lightning implementation
----------------------------------------

*  encoders.py         → NodeEncoder / EdgeEncoder
*  layers.py           → MessageBlock (small & large graphs)
*  heads.py            → DeepHead (H, S, D)
*  sparse_math.trace_* → Energy / electron count losses
"""

from __future__ import annotations
from itertools import product
from typing import Dict, Tuple, Any

import torch
import pytorch_lightning as pl
from torch import nn
from torch_scatter import scatter_add

from collections import OrderedDict

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_snap
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix, IrrepsBlockData
from data.graph_features import compute_graph_features, _minimal_disp

from net.common import Config, build_hidden_irreps
from net.encoders import NodeEncoder, EdgeEncoder
from net.layers import MessageBlock
from net.heads import DeepHead

# DeepH-E3
import sys
from pathlib import Path

# Add DeepH-E3 to the Python path
deeph_path = Path(__file__).resolve().parents[2] / "external" / "DeepH-E3"
if str(deeph_path) not in sys.path:
    sys.path.append(str(deeph_path))

from deephe3.model import Net as DeepHE3Net
from deephe3.e3modules import e3TensorDecomp, Rotate
from deephe3.utils import MaskMSELoss
from torch_geometric.data import Data, Batch


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
            # ! Improve this
            for key in t.keys():
                if key not in p.keys():
                    continue

                t_items_key = t_items[key]
                p_items_key = p_items[key]

                # Ensure edge order matches for comparison
                t_edges = t.pair_edges[key].t().tolist()
                p_edges = p.pair_edges[key].t().tolist()
                pred_edge_to_idx = {tuple(edge): i for i, edge in enumerate(p_edges)}

                target_indices = []
                pred_indices = []
                for i, edge in enumerate(t_edges):
                    if tuple(edge) in pred_edge_to_idx:
                        target_indices.append(i)
                        pred_indices.append(pred_edge_to_idx[tuple(edge)])

                if not target_indices:
                    continue

                target_blocks_to_compare = t_items_key[
                    torch.tensor(target_indices, device=self.device)
                ]
                pred_blocks_to_compare = p_items_key[
                    torch.tensor(pred_indices, device=self.device)
                ]

                mse_val += self._mse(pred_blocks_to_compare, target_blocks_to_compare)
                mae_val += torch.mean(
                    torch.abs(pred_blocks_to_compare - target_blocks_to_compare)
                )

            matrix_mses[name] = mse_val
            matrix_maes[name] = mae_val
            metrics[f"{stage}/{name}_mae"] = mae_val
            metrics[f"{stage}/{name}_mse"] = mse_val

            combined_matrix_losses[name] = (
                1 - self.cfg.loss_l1_fraction
            ) * mse_val + self.cfg.loss_l1_fraction * mae_val

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
            E_pred = trace_matmul_sparse_snap(
                preds_matrix["hamiltonian"], preds_matrix["density"]
            )
            E_true = y["energy"]
            metrics[f"{stage}/energy_mae"] = torch.mean(torch.abs(E_pred - E_true))
            if self.cfg.train_on_energy and not self.cfg.train_observables_on_gt:
                loss_E_weighted = self.cfg.loss_coef_observables * self._mse(
                    E_pred, E_true
                )

        if (
            self.cfg.enable_num_electrons
            and "overlap" in preds_matrix
            and "density" in preds_matrix
        ):
            N_pred = trace_matmul_sparse_snap(
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

            E_gt_D = trace_matmul_sparse_snap(preds_matrix["hamiltonian"], D_true)
            E_gt_H = trace_matmul_sparse_snap(H_true, preds_matrix["density"])
            N_gt_S = trace_matmul_sparse_snap(preds_matrix["density"], S_true)
            N_gt_D = trace_matmul_sparse_snap(S_true, preds_matrix["density"])

            if self.cfg.log_partial_gt_observables:
                metrics[f"{stage}/energy_mae_gt_density"] = torch.mean(
                    torch.abs(E_gt_D - E_true)
                )
                metrics[f"{stage}/energy_mae_gt_hamiltonian"] = torch.mean(
                    torch.abs(E_gt_H - E_true)
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


class DeepHE3(pl.LightningModule):
    def __init__(self, cfg: Config, mapper: BlockIrrepMapper):
        super().__init__()
        self.cfg = cfg
        self.mapper = mapper  # Used to get orbital info

        # Hardcoded hyperparameters from DeepH-E3 defaults
        self.deephe3 = DeepHE3Net(
            num_species=len(self.mapper.orbital_cfg.elements()),
            irreps_embed_node="16x0e",
            irreps_edge_init="16x0e",
            irreps_sh="1x0e+1x1o+1x2e",
            irreps_mid_node="16x0e+16x0o+16x1e+16x1o",
            irreps_post_node="16x0e+16x0o",
            irreps_out_node="1x0e",
            irreps_mid_edge="16x0e+16x0o+16x1e+16x1o",
            irreps_post_edge="16x0e+16x0o+16x1e+16x1o+16x2e+16x2o",
            irreps_out_edge="16x0e+16x0o+16x1e+16x1o+16x2e+16x2o",
            num_block=self.cfg.num_block,
            r_max=self.cfg.r_max,
            use_sc=self.cfg.use_sc,
            no_parity=False,
            use_sbf=self.cfg.use_sbf,
            only_ij=False,
            num_basis=self.cfg.num_basis,
        )

        # These are needed for transformations
        self.rotate_kernel = Rotate(default_dtype_torch=torch.float32, spinful=False)

        # Correctly derive out_js_list for Si orbitals (2s2p1d -> l=[0,0,1,1,2])
        si_orbitals_l = [0, 0, 1, 1, 2]
        out_js_list = list(product(si_orbitals_l, si_orbitals_l))

        self.construct_kernel = e3TensorDecomp(
            net_irreps_out=None,
            out_js_list=out_js_list,
            default_dtype_torch=torch.float32,
            spinful=False,
        )
        self.criterion = MaskMSELoss()

    def _prepare_deeph_input(self, x: Dict[str, Any]) -> Batch:
        num_nodes = x["positions"].shape[0]

        # Calculate displacement vectors and distances
        disp = _minimal_disp(x["positions"], x["edge_index"], x["box"])
        lengths = torch.linalg.norm(disp, dim=-1, keepdim=True)
        edge_attr = torch.cat([lengths, disp], dim=-1)

        data = Data(
            x=x["node_type_idx"],
            edge_index=x["edge_index"],
            edge_attr=edge_attr,
            pos=x["positions"],
            lattice=x["box"].unsqueeze(0),
            num_nodes=num_nodes,
        )
        return Batch.from_data_list([data])

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        batch = self._prepare_deeph_input(x)
        _, edge_fea = self.deephe3(batch)
        return edge_fea

    def _prepare_deeph_target(
        self, y_matrix: BlockMatrix, x: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a Mandala BlockMatrix into the flat label and mask format for DeepH-E3."""
        num_edges = x["edge_index"].shape[1]
        # For Si 2s2p1d, the blocks are 13x13, so flattened size is 169
        label = torch.zeros(num_edges, 169, device=self.device)
        mask = torch.zeros_like(label, dtype=torch.bool)

        edge_to_block = {
            tuple(edge.tolist()): block
            for edge, block in zip(
                y_matrix.pair_edges["Si-Si"].t(), y_matrix.pair_blocks["Si-Si"]
            )
        }

        for i, edge in enumerate(x["edge_index"].t().tolist()):
            if tuple(edge) in edge_to_block:
                label[i] = edge_to_block[tuple(edge)].flatten()
                mask[i] = True

        return label, mask

    def _shared_step(self, batch, batch_idx, stage: str):
        x, y = batch
        predicted_irreps = self.forward(x)

        # Reconstruct predicted matrix in OpenMX basis
        H_pred_openmx_flat = self.construct_kernel.get_H(predicted_irreps)

        # Prepare target matrix in the same flat, OpenMX format
        H_true_openmx_flat, mask = self._prepare_deeph_target(y["hamiltonian"], x)

        loss = self.criterion(H_pred_openmx_flat, H_true_openmx_flat, mask)
        mae = torch.mean(torch.abs(H_pred_openmx_flat[mask] - H_true_openmx_flat[mask]))

        self.log(f"{stage}/loss", loss)
        self.log(f"{stage}/hamiltonian_mae", mae)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return optimizer
