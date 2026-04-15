"""
E3GNN – PyTorch-Lightning implementation
----------------------------------------

*  encoders.py         -> NodeEncoder / EdgeEncoder
*  layers.py           -> MessageBlock (small & large graphs)
*  heads.py            -> DeepHead (H, S, D)
*  sparse_math.trace_* -> Energy / electron count losses
"""

from __future__ import annotations
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from torch import nn

from collections import OrderedDict

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import (
    trace_matmul_sparse_block_matrix_aligned,
)
from data.snapshot import Snapshot
from data.block_matrix import IrrepsBlockData
from data.graph_features import compute_edge_geometry_from_static_edges

from net.common import Config, resolve_hidden_irreps
from net.irrep_tools import (
    build_irrep_projector_cache,
    compute_irrep_metrics,
    get_all_irreps,
    project_irrep_vectors_to_blocks,
)
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
        self.save_hyperparameters(ignore=["mapper"])
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
        if cfg.partial_train not in (None, "diag", "shifted_self", "offdiag"):
            raise ValueError(
                "partial_train must be one of None, 'diag', 'shifted_self', or 'offdiag'."
            )
        if cfg.train_on_irrep_parts and cfg.train_target == "irreps":
            raise ValueError(
                "train_on_irrep_parts expects matrix-space supervision and should not be combined with train_target='irreps'."
            )

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = resolve_hidden_irreps(self.cfg)
        self.neck_irreps: Irreps = resolve_hidden_irreps(self.cfg)
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max)
        self._compiled_forward_core = None

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
        self.mp_blocks = nn.ModuleList()
        node_irreps = self.node_enc.irreps_out
        edge_irreps = self.edge_enc.irreps_out
        for i in range(self.cfg.num_layers_gnn):
            block = MessageBlock(
                node_irreps=node_irreps,
                edge_irreps=edge_irreps,
                node_irreps_out=self.hidden_irreps,
                edge_irreps_out=self.hidden_irreps,
                num_species=len(self.mapper.orbital_cfg.elements()),
                cfg=self.cfg,
                info={"layer": i},
            )
            self.mp_blocks.append(block)
            node_irreps = block.node_irreps_out
            edge_irreps = block.edge_irreps_out
        self.final_node_irreps = node_irreps
        self.final_edge_irreps = edge_irreps
        self.all_irreps = get_all_irreps(self.mapper)
        self.irrep_projectors = build_irrep_projector_cache(
            self.mapper, self.all_irreps
        )

        # ---------- heads ----------------------------------------------
        pair_keys = list(
            map(lambda pair: f"{pair[0]}-{pair[1]}", self.mapper._maps.keys())
        )
        self.heads = nn.ModuleDict(
            {
                name: DeepHead(
                    irreps_diag_in=(
                        self.final_node_irreps
                        if self.cfg.head_use_node_embeddings_for_self_edges
                        else self.final_edge_irreps
                    ),
                    irreps_edge_in=self.final_edge_irreps,
                    irreps_neck=self.neck_irreps,
                    pair_keys=pair_keys,
                    mapper=self.mapper,
                    cfg=self.cfg,
                    info={"matrix": name},
                )
                for name in self.cfg.matrix_targets
            }
        )

        self._apply_init_weights_factor()
        self._setup_compiled_forward()

        # Print model summary if verbosity >= 1
        print_model_summary(self, verbosity=self.cfg.verbosity)

    @classmethod
    def load_from_checkpoint_with_mapper(
        cls,
        checkpoint_path: str,
        mapper: BlockIrrepMapper,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
        **kwargs,
    ) -> "E3GNN":
        return cls.load_from_checkpoint(
            checkpoint_path,
            mapper=mapper,
            map_location=map_location,
            strict=strict,
            **kwargs,
        )

    def _apply_init_weights_factor(self) -> None:
        factor = float(self.cfg.init_weights_factor)
        if factor == 1.0:
            return
        with torch.no_grad():
            for param in self.parameters():
                if param.is_floating_point():
                    param.mul_(factor)

    def _setup_compiled_forward(self) -> None:
        if not self.cfg.compile_model:
            return
        if self.cfg.log_activation_mag:
            raise ValueError(
                "compile_model=True is not supported together with log_activation_mag=True."
            )
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "compile_model=True requested, but torch.compile is not available."
            )
        self._compiled_forward_core = torch.compile(
            self._forward_core,
            mode=self.cfg.compile_mode,
            fullgraph=bool(self.cfg.compile_fullgraph),
            dynamic=False,
        )

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
        raw: Dict[str, torch.Tensor],
        x: Dict[str, Any],
    ) -> IrrepsBlockData:
        """
        Convert DeepHead raw dict -> IrrepsBlockData with mapper.
        """
        irreps_blocks = IrrepsBlockData(
            atoms=x["atoms_tuple"],
            atom_counts=x["atom_counts"],
            pair_vectors=raw,
            pair_edges=x["pred_pair_edges_static"],
            lookup=x["pred_lookup_static"],
            orbital_cfg=self.mapper.orbital_cfg,
        )
        return irreps_blocks

    def _should_recompute_edge_features(self, x: Dict[str, Any]) -> bool:
        if not self.cfg.precompute_edge_features:
            return True
        if "positions" in x and x["positions"].requires_grad:
            return True
        if x.get("box") is not None and x["box"].requires_grad:
            return True
        return False

    def _populate_edge_features(self, x: Dict[str, Any]) -> None:
        edge_length_emb, edge_sh, _ = compute_edge_geometry_from_static_edges(
            positions=x["positions"],
            box=x["box"],
            edge_index=x["edge_index"],
            edge_shift=x["edge_shift"],
            sh_irreps=self.sh_irreps,
            cutoff_radius=self.cfg.cutoff_radius,
            n_radial=self.cfg.n_radial,
            radial_embedding_scale=self.cfg.radial_embedding_scale,
        )
        x["edge_length_emb"] = edge_length_emb
        x["edge_sh"] = edge_sh

    def _partial_train_mask(self, edges_5d: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(edges_5d.shape[1], dtype=torch.bool, device=edges_5d.device)
        if self.cfg.partial_train is None:
            return mask

        sx, sy, sz, i, j = edges_5d
        is_diag = (i == j) & (sx == 0) & (sy == 0) & (sz == 0)
        is_shifted_self = (i == j) & ~((sx == 0) & (sy == 0) & (sz == 0))

        if self.cfg.partial_train == "diag":
            return is_diag
        if self.cfg.partial_train == "shifted_self":
            return is_shifted_self
        if self.cfg.partial_train == "offdiag":
            if self.cfg.separate_shifted_self:
                return i != j
            return ~is_diag
        raise RuntimeError(f"Unsupported partial_train value: {self.cfg.partial_train}")

    def _compute_irrep_part_losses(
        self,
        pred: IrrepsBlockData,
        target: IrrepsBlockData,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        total_mse = torch.tensor(0.0, device=self.device)
        total_mae = torch.tensor(0.0, device=self.device)
        per_irrep: dict[str, dict[str, torch.Tensor]] = {}

        for irrep in self.all_irreps:
            irrep_key = str(irrep)
            irrep_mse = torch.tensor(0.0, device=self.device)
            irrep_mae = torch.tensor(0.0, device=self.device)
            saw_any = False

            for pair_key, projector in self.irrep_projectors[irrep_key].items():
                if (
                    pair_key not in pred.pair_vectors
                    or pair_key not in target.pair_vectors
                ):
                    continue

                pred_blocks = project_irrep_vectors_to_blocks(
                    pred.pair_vectors[pair_key], projector, self.mapper
                )
                target_blocks = project_irrep_vectors_to_blocks(
                    target.pair_vectors[pair_key], projector, self.mapper
                )

                min_n = min(pred_blocks.shape[0], target_blocks.shape[0])
                if min_n <= 0:
                    continue

                pred_edges = pred.pair_edges[pair_key][:, :min_n]
                target_edges = target.pair_edges[pair_key][:, :min_n]
                if self.cfg.safety_checks:
                    assert torch.equal(
                        pred_edges, target_edges
                    ), f"Edge mismatch in irrep-part loss for key {pair_key}, irrep {irrep_key}."

                partial_mask = self._partial_train_mask(target_edges)
                if not partial_mask.any():
                    continue

                pred_selected = pred_blocks[:min_n][partial_mask]
                target_selected = target_blocks[:min_n][partial_mask]
                if pred_selected.shape[0] == 0:
                    continue

                irrep_mse = irrep_mse + self._mse(pred_selected, target_selected)
                irrep_mae = irrep_mae + self._mae(pred_selected, target_selected)
                saw_any = True

            if saw_any:
                per_irrep[irrep_key] = {"mse": irrep_mse, "mae": irrep_mae}
                total_mse = total_mse + irrep_mse
                total_mae = total_mae + irrep_mae

        return total_mse, total_mae, per_irrep

    def _forward_core(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self._should_recompute_edge_features(x):
            self._populate_edge_features(x)

        # ---- encode ----------------------------------------------------
        activation_mags = getattr(self, "_activation_mags", None)
        node = self.node_enc(x["node_type_idx"], activation_mags=activation_mags)
        edge = self.edge_enc(
            x["edge_type_idx"],
            x["edge_length_emb"],
            x["edge_sh"],
            activation_mags=activation_mags,
        )
        if self.cfg.safety_checks and torch.is_grad_enabled():
            assert node.requires_grad, "Gradients not flowing through node encoder!"
            assert edge.requires_grad, "Gradients not flowing through edge encoder!"

        # Single unified message-passing loop
        for idx, blk in enumerate(self.mp_blocks):
            node, edge = blk(
                node=node,
                edge=edge,
                edge_index=x["edge_index"],
                edge_sh=x["edge_sh"],
                edge_length_emb=x["edge_length_emb"],
                node_one_hot=x["node_one_hot"],
                edge_one_hot=x["edge_one_hot"],
                activation_mags=activation_mags,
            )

        return {
            name: head(node, edge, x["pred_pair_edges_static"], x["edge_partitions"])
            for name, head in self.heads.items()
        }

    # ------------------------------------------------------------------ forward
    def forward(self, x: Dict[str, Any]):
        # initialize activation magnitudes storage
        self._activation_mags: dict[str, torch.Tensor] = OrderedDict()

        preds_raw = (
            self._compiled_forward_core(x)
            if self._compiled_forward_core is not None
            else self._forward_core(x)
        )

        preds_wrapped = {
            name: self._wrap_head_output(raw, x) for name, raw in preds_raw.items()
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
        pred_trace_alignment = x["pred_trace_alignment"]

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
        # num_atoms = x["positions"].shape[0]

        for name in self.cfg.matrix_targets:
            per_irrep_metrics: dict[str, dict[str, torch.Tensor]] = {}

            if self.cfg.train_on_irrep_parts:
                target_irreps = y[name].to_vectors(self.mapper)
                mse_val, mae_val, per_irrep_metrics = self._compute_irrep_part_losses(
                    preds_irreps[name], target_irreps
                )
            else:
                p = (
                    preds_irreps[name]
                    if self.cfg.train_target == "irreps"
                    else preds_matrix[name]
                )
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

                    min_n = min(preds.shape[0], targets.shape[0])
                    preds = preds[:min_n]
                    targets = targets[:min_n]
                    pred_edges = p.pair_edges[key][:, :min_n]
                    target_edges = t.pair_edges[key][:, :min_n]

                    if self.cfg.safety_checks:
                        assert torch.equal(
                            pred_edges, target_edges
                        ), f"Edge mismatch in block loss for matrix {name}, key {key}."

                    partial_mask = self._partial_train_mask(target_edges)
                    preds = preds[partial_mask]
                    targets = targets[partial_mask]
                    if preds.shape[0] == 0:
                        continue

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
            if self.cfg.log_per_irrep_metrics or self.cfg.train_on_irrep_parts:
                for irrep_key, irrep_vals in per_irrep_metrics.items():
                    irrep_mse = irrep_vals["mse"]
                    irrep_mae = irrep_vals["mae"]
                    irrep_loss = (
                        1 - self.cfg.loss_l1_fraction
                    ) * irrep_mse + self.cfg.loss_l1_fraction * irrep_mae
                    if name == "hamiltonian":
                        metrics[f"{stage}/{name}_irrep_{irrep_key}_mae"] = (
                            irrep_mae * HARTREE_TO_EV
                        )
                        metrics[f"{stage}/{name}_irrep_{irrep_key}_mse"] = irrep_mse * (
                            HARTREE_TO_EV**2
                        )
                    else:
                        metrics[f"{stage}/{name}_irrep_{irrep_key}_mae"] = irrep_mae
                        metrics[f"{stage}/{name}_irrep_{irrep_key}_mse"] = irrep_mse
                    if self.cfg.train_on_irrep_parts:
                        metrics[f"{stage}/{name}_irrep_{irrep_key}_loss"] = irrep_loss

            combined_matrix_losses[name] = (
                1 - self.cfg.loss_l1_fraction
            ) * mse_for_loss + self.cfg.loss_l1_fraction * mae_for_loss

        loss_matrix = sum(combined_matrix_losses.values())

        # --- Observable Evaluation --------------------------------------
        t_obs_start = time.perf_counter()
        loss_E_weighted = torch.tensor(0.0, device=self.device)
        loss_N_weighted = torch.tensor(0.0, device=self.device)
        loss_F_weighted = torch.tensor(0.0, device=self.device)

        H_true = None
        D_true = None
        S_true = None
        needs_gt_observables = (
            self.cfg.enable_energy
            or self.cfg.enable_num_electrons
            or self.cfg.log_partial_gt_observables
            or self.cfg.train_observables_on_gt
        )
        if needs_gt_observables and {"hamiltonian", "density", "overlap"}.issubset(y):
            if self.cfg.train_target == "irreps":
                H_true = y["hamiltonian"].to_blocks(self.mapper)
                D_true = y["density"].to_blocks(self.mapper)
                S_true = y["overlap"].to_blocks(self.mapper)
            else:
                H_true = y["hamiltonian"]
                D_true = y["density"]
                S_true = y["overlap"]

        # Standard observables
        if (
            self.cfg.enable_energy
            and "hamiltonian" in preds_matrix
            and "density" in preds_matrix
        ):
            E_pred = trace_matmul_sparse_block_matrix_aligned(
                preds_matrix["hamiltonian"],
                preds_matrix["density"],
                pred_trace_alignment,
            )
            E_true = y["energy"]
            metrics[f"{stage}/energy_mae"] = (
                torch.mean(torch.abs(E_pred - E_true)) * HARTREE_TO_EV
            )
            if H_true is not None:
                E_gt_H = trace_matmul_sparse_block_matrix_aligned(
                    H_true, preds_matrix["density"], pred_trace_alignment
                )
                metrics[f"{stage}/energy_mae_gt_hamiltonian"] = (
                    torch.mean(torch.abs(E_gt_H - E_true)) * HARTREE_TO_EV
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
            N_pred = trace_matmul_sparse_block_matrix_aligned(
                preds_matrix["density"],
                preds_matrix["overlap"],
                pred_trace_alignment,
            )
            N_true = y["num_electrons"]
            metrics[f"{stage}/num_electrons_mae"] = torch.mean(
                torch.abs(N_pred - N_true)
            )
            if self.cfg.train_on_num_electrons and not self.cfg.train_observables_on_gt:
                loss_N_weighted = self.cfg.loss_coef_observables * self._mse(
                    N_pred, N_true
                )

        if self.cfg.enable_forces and y.get("forces") is not None:
            forces_pred = self.get_forces(preds_irreps, x["positions"], x["box"])
            forces_true = y["forces"]
            metrics[f"{stage}/forces_mae"] = torch.mean(
                torch.abs(forces_pred - forces_true)
            )
            metrics[f"{stage}/forces_mse"] = self._mse(forces_pred, forces_true)
            if self.cfg.train_on_forces:
                loss_F_weighted = self.cfg.loss_coef_forces * self._mse(
                    forces_pred, forces_true
                )

        if self.cfg.log_per_irrep_metrics:
            for name in self.cfg.matrix_targets:
                if name not in preds_irreps or name not in y:
                    continue
                target_irreps = (
                    y[name]
                    if self.cfg.train_target == "irreps"
                    else y[name].to_vectors(self.mapper)
                )
                irrep_metrics = compute_irrep_metrics(
                    preds_irreps[name],
                    target_irreps,
                    self.all_irreps,
                    self.mapper,
                )
                for key, value in irrep_metrics.items():
                    metrics[f"{stage}/{name}_irrep_{key}"] = value

        # Partial Ground Truth Observables
        if self.cfg.log_partial_gt_observables or self.cfg.train_observables_on_gt:
            E_gt_D = trace_matmul_sparse_block_matrix_aligned(
                preds_matrix["hamiltonian"], D_true, pred_trace_alignment
            )
            E_gt_H = trace_matmul_sparse_block_matrix_aligned(
                H_true, preds_matrix["density"], pred_trace_alignment
            )
            N_gt_S = trace_matmul_sparse_block_matrix_aligned(
                preds_matrix["density"], S_true, pred_trace_alignment
            )
            N_gt_D = trace_matmul_sparse_block_matrix_aligned(
                S_true, preds_matrix["density"], pred_trace_alignment
            )

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
        total_l2_loss = (
            total_matrix_l2_component
            + loss_E_weighted
            + loss_N_weighted
            + loss_F_weighted
        )

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
        metrics[f"{stage}/loss_block_total"] = loss_matrix

        if loss_E_weighted > 0:
            metrics[f"{stage}/loss_energy"] = loss_E_weighted
            metrics[f"{stage}/loss_energy_weighted"] = loss_E_weighted
        if loss_N_weighted > 0:
            metrics[f"{stage}/loss_num_electrons"] = loss_N_weighted
            metrics[f"{stage}/loss_num_electrons_weighted"] = loss_N_weighted
        if loss_F_weighted > 0:
            metrics[f"{stage}/loss_forces"] = loss_F_weighted
            metrics[f"{stage}/loss_forces_weighted"] = loss_F_weighted

        for name, loss_val in combined_matrix_losses.items():
            metrics[f"{stage}/loss_block_{name}"] = loss_val

        if stage == "train" and loss > 1e-12:
            for name, val in combined_matrix_losses.items():
                metrics[f"frac/loss_{name}"] = val / loss
            if loss_E_weighted > 0:
                metrics["frac/loss_energy"] = loss_E_weighted / loss
            if loss_N_weighted > 0:
                metrics["frac/loss_num_electrons"] = loss_N_weighted / loss
            if loss_F_weighted > 0:
                metrics["frac/loss_forces"] = loss_F_weighted / loss

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

    def on_before_optimizer_step(self, optimizer) -> None:
        total_norm_sq = torch.tensor(0.0, device=self.device)
        saw_grad = False
        for param in self.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_norm_sq = total_norm_sq + torch.sum(grad * grad)
            saw_grad = True
        if saw_grad:
            self.log(
                "grad_norm",
                torch.sqrt(total_norm_sq),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=1,
            )

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
        # 4. optionally symmetrize: σ -> (σ+σ^T)/2
        if symmetrize:
            stress = 0.5 * (stress + stress.transpose(-1, -2))
        return stress

    def predict_forces(self, x: Dict[str, Any]) -> torch.Tensor:
        x = dict(x)
        if not x["positions"].requires_grad:
            x["positions"] = x["positions"].clone().detach().requires_grad_(True)
        predictions = self(x)
        return self.get_forces(predictions, x["positions"], x["box"])

    def predict_stress(self, x: Dict[str, Any]) -> torch.Tensor:
        x = dict(x)
        if x.get("box") is not None and not x["box"].requires_grad:
            x["box"] = x["box"].clone().detach().requires_grad_(True)
        predictions = self(x)
        return self.get_stress(predictions, x["positions"], x["box"])
