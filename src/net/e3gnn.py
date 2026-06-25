"""
E3GNN - PyTorch-Lightning implementation
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
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix, IrrepsBlockData
from data.envelope import scale_block_matrix_by_edge_values
from data.graph_features import compute_edge_geometry_from_static_edges

from net.common import Config, resolve_hidden_irreps
from net.irrep_tools import (
    build_irrep_projector_cache,
    build_irrep_block_matrix_cache,
    compute_irrep_metrics,
    compute_hamiltonian_mae_contributions,
    get_all_irreps,
    project_irrep_vectors_to_blocks,
)
from net.observable_metrics import (
    add_observable_metrics,
    build_observable_predictions,
    observable_loss,
    truncate_pred_block_matrix_to_target_prefix,
    validate_observable_config,
)
from net.spectral_loss import compute_spectral_eigenvalue_loss
from net.checkpoint_compat import strip_mapper_keys
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
        if cfg.train_on_irrep_parts and cfg.train_target != "matrix":
            raise ValueError(
                "train_on_irrep_parts expects matrix-space supervision and should only be used with train_target='matrix'."
            )
        validate_observable_config(cfg)
        envelope_mode = str(getattr(cfg, "hamiltonian_envelope_mode", "off")).lower()
        if envelope_mode not in {"off", "normalize_target", "multiply_prediction"}:
            raise ValueError(
                "hamiltonian_envelope_mode must be one of 'off', "
                "'normalize_target', or 'multiply_prediction'."
            )
        if bool(getattr(cfg, "pair_conditioned_radial_mlp", False)) and bool(
            getattr(cfg, "train_on_irrep_parts", False)
        ):
            raise ValueError(
                "pair_conditioned_radial_mlp is not supported together with "
                "train_on_irrep_parts."
            )
        if bool(getattr(cfg, "train_on_irrep_parts", False)) and envelope_mode != "off":
            raise ValueError(
                "hamiltonian_envelope_mode requires matrix-space training; "
                "it cannot be used with train_on_irrep_parts."
            )
        if bool(
            getattr(cfg, "spectral_loss_enabled", False)
        ) and "hamiltonian" not in set(cfg.matrix_targets):
            raise ValueError(
                "spectral_loss_enabled requires 'hamiltonian' to be present in matrix_targets."
            )

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = resolve_hidden_irreps(self.cfg)
        self.neck_irreps: Irreps = resolve_hidden_irreps(self.cfg)
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
                n_edge_types=len(self.mapper.edge_types),
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

        self._nan_loss_detected = False
        self._spectral_loss_debug_printed = False
        self._apply_init_weights_factor()
        self._apply_trainable_parameter_freeze()

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

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        return super().load_state_dict(strip_mapper_keys(state_dict), strict=strict)

    def _apply_init_weights_factor(self) -> None:
        factor = float(self.cfg.init_weights_factor)
        if factor == 1.0:
            return
        with torch.no_grad():
            for param in self.parameters():
                if param.is_floating_point():
                    param.mul_(factor)

    def _apply_trainable_parameter_freeze(self) -> None:
        if not bool(getattr(self.cfg, "freeze_backbone_train_heads_only", False)):
            return
        frozen_modules = [self.node_enc, self.edge_enc, self.mp_blocks]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(
            "--- Head-only fine-tuning enabled: "
            f"trainable_params={trainable}, frozen_params={frozen} ---"
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

    def _matrix_envelope_mode(self) -> str:
        return str(getattr(self.cfg, "hamiltonian_envelope_mode", "off")).lower()

    def _physicalize_predicted_matrix(
        self,
        *,
        name: str,
        pred_matrix: BlockMatrix,
        x: Dict[str, Any],
    ) -> BlockMatrix:
        mode = self._matrix_envelope_mode()
        if name not in {"hamiltonian", "overlap"} or mode == "off":
            return pred_matrix
        if "edge_envelope" not in x:
            raise ValueError(
                "hamiltonian_envelope_mode requires edge_envelope in the batch."
            )
        if "edge_partitions" not in x:
            raise ValueError(
                "hamiltonian_envelope_mode requires edge_partitions in the batch."
            )
        if mode in {"multiply_prediction", "normalize_target"}:
            return scale_block_matrix_by_edge_values(
                pred_matrix,
                x["edge_envelope"],
                x["edge_partitions"],
            )
        raise ValueError(
            "hamiltonian_envelope_mode must be one of 'off', "
            "'normalize_target', or 'multiply_prediction'."
        )

    def predicted_irreps_to_block_matrices(
        self,
        predictions: Dict[str, IrrepsBlockData],
        x: Dict[str, Any],
        *,
        physical: bool = True,
    ) -> Dict[str, BlockMatrix]:
        matrices = {
            name: pred.to_blocks(self.mapper) for name, pred in predictions.items()
        }
        if not physical:
            return matrices
        return {
            name: self._physicalize_predicted_matrix(
                name=name,
                pred_matrix=matrix,
                x=x,
            )
            for name, matrix in matrices.items()
        }

    def _spectral_loss_enabled(self) -> bool:
        return bool(getattr(self.cfg, "spectral_loss_enabled", False))

    def _compute_spectral_loss(
        self,
        *,
        x: Dict[str, Any],
        physical_pred_hamiltonian: BlockMatrix,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        required = {
            "spectral_kpoints_abs",
            "spectral_shifts",
            "spectral_gt_overlap_k",
            "spectral_gt_eigs_ev",
            "spectral_gt_fermi_ev",
            "spectral_window_weights",
        }
        missing = sorted(key for key in required if key not in x)
        if missing:
            raise ValueError(
                "spectral_loss_enabled requires cached spectral payload in the batch; "
                f"missing keys: {missing}"
            )
        if not self._spectral_loss_debug_printed:
            print(
                "--- Running first spectral loss evaluation: "
                f"kpoints={int(x['spectral_kpoints_abs'].shape[0])}, "
                f"window_weight_sum={float(x['spectral_window_weights'].sum().item()):.1f}, "
                f"huber_delta_ev={float(self.cfg.spectral_loss_huber_delta_ev):.3f} ---",
                flush=True,
            )
        loss, stats = compute_spectral_eigenvalue_loss(
            pred_hamiltonian=physical_pred_hamiltonian,
            spectral_payload=x,
            box=x["box"],
            huber_delta_ev=float(self.cfg.spectral_loss_huber_delta_ev),
            overlap_psd_cleanup=bool(
                getattr(self.cfg, "spectral_loss_overlap_psd_cleanup", False)
            ),
            overlap_jitter=bool(
                getattr(self.cfg, "spectral_loss_overlap_jitter", True)
            ),
        )
        if not self._spectral_loss_debug_printed:
            print(
                "--- First spectral loss evaluation finished: "
                f"loss={float(loss.detach().item()):.6f}, "
                f"spectral_mae_ev={float(stats['spectral_mae_ev'].detach().item()):.6f} ---",
                flush=True,
            )
            self._spectral_loss_debug_printed = True
        return loss, stats

    def _apply_matrix_envelope_mode(
        self,
        *,
        name: str,
        pred_matrix: BlockMatrix,
        target_matrix: BlockMatrix,
        x: Dict[str, Any],
    ) -> tuple[BlockMatrix, BlockMatrix, BlockMatrix]:
        mode = self._matrix_envelope_mode()
        if name not in {"hamiltonian", "overlap"} or mode == "off":
            return pred_matrix, target_matrix, pred_matrix
        if "edge_envelope" not in x:
            raise ValueError(
                "hamiltonian_envelope_mode requires edge_envelope in the batch."
            )
        if "edge_partitions" not in x:
            raise ValueError(
                "hamiltonian_envelope_mode requires edge_partitions in the batch."
            )
        edge_envelope = x["edge_envelope"]
        edge_partitions = x["edge_partitions"]
        eps = float(getattr(self.cfg, "hamiltonian_envelope_eps", 1.0e-12))
        if mode == "multiply_prediction":
            scaled_pred = scale_block_matrix_by_edge_values(
                pred_matrix,
                edge_envelope,
                edge_partitions,
            )
            return scaled_pred, target_matrix, scaled_pred
        if mode == "normalize_target":
            physical_pred = scale_block_matrix_by_edge_values(
                pred_matrix,
                edge_envelope,
                edge_partitions,
            )
            normalized_target = scale_block_matrix_by_edge_values(
                target_matrix,
                edge_envelope,
                edge_partitions,
                inverse=True,
                eps=eps,
                allow_prefix_trim=True,
            )
            return pred_matrix, normalized_target, physical_pred
        raise ValueError(
            "hamiltonian_envelope_mode must be one of 'off', "
            "'normalize_target', or 'multiply_prediction'."
        )

    def _edge_loss_weights(self, x: Dict[str, Any]) -> torch.Tensor | None:
        mode = str(getattr(self.cfg, "loss_weighting_mode", "off")).lower()
        if mode == "off":
            return None
        if "edge_partitions" not in x:
            raise ValueError(
                "loss_weighting_mode requires edge_partitions in the batch."
            )
        if mode in {"envelope_inverse_sqrt_clipped", "envelope_inverse_clipped"}:
            if "edge_envelope" not in x:
                raise ValueError(
                    "loss_weighting_mode based on envelope requires edge_envelope in the batch."
                )
            base = x["edge_envelope"]
            eps = float(getattr(self.cfg, "hamiltonian_envelope_eps", 1.0e-12))
            if mode == "envelope_inverse_sqrt_clipped":
                weights = torch.rsqrt(base.clamp_min(eps))
            else:
                weights = base.clamp_min(eps).reciprocal()
        elif mode == "distance_short_range_bias":
            if "edge_length" not in x:
                raise ValueError(
                    "distance_short_range_bias loss weighting requires edge_length in the batch."
                )
            weights = torch.rsqrt(1.0 + x["edge_length"].clamp_min(0.0))
        else:
            raise ValueError(
                "loss_weighting_mode must be one of 'off', "
                "'distance_short_range_bias', "
                "'envelope_inverse_sqrt_clipped', or 'envelope_inverse_clipped'."
            )

        min_w = float(getattr(self.cfg, "loss_weight_min", 0.0))
        max_w = float(getattr(self.cfg, "loss_weight_max", 1.0))
        if max_w < min_w:
            raise ValueError("loss_weight_max must be >= loss_weight_min.")
        return weights.clamp(min=min_w, max=max_w)

    def _weighted_edge_block_losses(
        self,
        *,
        preds: torch.Tensor,
        targets: torch.Tensor,
        edge_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if preds.shape != targets.shape:
            raise ValueError(
                "Predicted and target block tensors must have identical shapes, got "
                f"{tuple(preds.shape)} and {tuple(targets.shape)}"
            )
        if preds.ndim != 3:
            raise ValueError(
                f"Block tensors must have shape (E, D1, D2), got {tuple(preds.shape)}"
            )

        per_edge_mse = torch.mean((preds - targets) ** 2, dim=(1, 2))
        per_edge_mae = torch.mean(torch.abs(preds - targets), dim=(1, 2))
        if edge_weights is None:
            return per_edge_mse.mean(), per_edge_mae.mean()

        if edge_weights.ndim != 1 or edge_weights.shape[0] != per_edge_mse.shape[0]:
            raise ValueError(
                "Edge weights must be a 1D tensor matching the number of selected "
                f"edges, got weights={tuple(edge_weights.shape)} and "
                f"selected_edges={int(per_edge_mse.shape[0])}"
            )
        edge_weights = edge_weights.to(device=preds.device, dtype=preds.dtype)
        weight_sum = edge_weights.sum()
        if not torch.isfinite(weight_sum) or float(weight_sum) <= 0.0:
            raise ValueError("Edge weights must have a positive finite sum.")
        weighted_mse = torch.sum(edge_weights * per_edge_mse) / weight_sum
        weighted_mae = torch.sum(edge_weights * per_edge_mae) / weight_sum
        return weighted_mse, weighted_mae

    def _should_recompute_edge_features(self, x: Dict[str, Any]) -> bool:
        if not self.cfg.precompute_edge_features:
            return True
        if "positions" in x and x["positions"].requires_grad:
            return True
        if x.get("box") is not None and x["box"].requires_grad:
            return True
        return False

    def _populate_edge_features(self, x: Dict[str, Any]) -> None:
        radial_lengths = None
        radial_basis_end = None
        if (
            str(getattr(self.cfg, "pair_distance_normalization", "off")).lower()
            == "pair_r0"
        ):
            if "edge_length" not in x or "edge_r0" not in x:
                raise ValueError(
                    "pair_distance_normalization='pair_r0' requires edge_length "
                    "and edge_r0 in the batch."
                )
            radial_lengths = x["edge_length"] / x["edge_r0"].clamp_min(1e-12)
            radial_basis_end = 1.0
        edge_length_emb, edge_sh, _ = compute_edge_geometry_from_static_edges(
            positions=x["positions"],
            box=x["box"],
            edge_index=x["edge_index"],
            edge_shift=x["edge_shift"],
            sh_irreps=self.sh_irreps,
            cutoff_radius=self.cfg.cutoff_radius,
            n_radial=self.cfg.n_radial,
            radial_embedding_scale=self.cfg.radial_embedding_scale,
            radial_lengths=radial_lengths,
            radial_basis_end=radial_basis_end,
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

    def _stop_training_on_nonfinite_loss(
        self,
        loss: torch.Tensor,
        *,
        stage: str,
        batch_idx: int,
        phase: str,
    ) -> bool:
        if stage != "train":
            return False
        if torch.isfinite(loss).all():
            return False

        self._nan_loss_detected = True
        print(
            "--- Non-finite train loss detected "
            f"(phase={phase}, batch_idx={batch_idx}); "
            "stopping before optimizer step and finishing evaluation. ---",
            flush=True,
        )
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            try:
                trainer.should_stop = True
            except Exception:
                pass
        return True

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

                target_n = target_blocks.shape[0]
                if target_n <= 0:
                    continue
                if pred_blocks.shape[0] < target_n:
                    raise ValueError(
                        f"Predicted irrep blocks for key {pair_key} and irrep {irrep_key} "
                        f"are too short: pred_len={pred_blocks.shape[0]} target_len={target_n}"
                    )

                pred_edges = pred.pair_edges[pair_key][:, :target_n]
                target_edges = target.pair_edges[pair_key][:, :target_n]
                if self.cfg.safety_checks:
                    assert torch.equal(
                        pred_edges, target_edges
                    ), f"Edge mismatch in irrep-part loss for key {pair_key}, irrep {irrep_key}."

                partial_mask = self._partial_train_mask(target_edges)
                if not partial_mask.any():
                    continue

                pred_selected = pred_blocks[:target_n][partial_mask]
                target_selected = target_blocks[:target_n][partial_mask]
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
                edge_type_idx=x["edge_type_idx"],
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

        preds_raw = self._forward_core(x)

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
        combined_pair_losses: dict[str, dict[str, torch.Tensor]] = {}
        hamiltonian_mae_contribs: dict[str, object] | None = None
        irrep_block_cache_by_name: dict[
            str, tuple[dict[str, BlockMatrix], dict[str, BlockMatrix]]
        ] = {}
        allow_train_metrics = stage != "train" or bool(self.cfg.log_train_metrics)
        edge_loss_weights = self._edge_loss_weights(x)
        # num_atoms = x["positions"].shape[0]

        def _compute_physical_matrix_metrics(
            pred_matrix: BlockMatrix,
            target_matrix: BlockMatrix,
            matrix_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            mse_val = torch.tensor(0.0, device=self.device)
            mae_val = torch.tensor(0.0, device=self.device)
            for key in target_matrix.pair_blocks.keys():
                if key not in pred_matrix.pair_blocks:
                    raise ValueError(f"Key {key} not found in predicted items.")
                preds = pred_matrix.pair_blocks[key]
                targets = target_matrix.pair_blocks[key]
                target_n = targets.shape[0]
                if target_n <= 0:
                    continue
                if preds.shape[0] < target_n:
                    raise ValueError(
                        f"Predicted blocks for matrix {matrix_name}, key {key} are too short: "
                        f"pred_len={preds.shape[0]} target_len={target_n}"
                    )
                preds = preds[:target_n]
                targets = targets[:target_n]
                pred_edges = pred_matrix.pair_edges[key][:, :target_n]
                target_edges = target_matrix.pair_edges[key][:, :target_n]
                if self.cfg.safety_checks:
                    assert torch.equal(
                        pred_edges, target_edges
                    ), f"Edge mismatch in metric block loss for matrix {matrix_name}, key {key}."
                partial_mask = self._partial_train_mask(target_edges)
                preds = preds[partial_mask]
                targets = targets[partial_mask]
                if preds.shape[0] == 0:
                    continue
                mse_val += self._mse(preds, targets)
                mae_val += self._mae(preds, targets)
            return mse_val, mae_val

        for name in self.cfg.matrix_targets:
            per_irrep_metrics: dict[str, dict[str, torch.Tensor]] = {}
            pred_irrep_blocks = None
            target_irrep_blocks = None
            pair_losses: dict[str, torch.Tensor] = {}
            raw_pred_matrix = preds_matrix[name]
            raw_target_matrix = y[name]
            loss_pred_matrix, loss_target_matrix, metric_pred_matrix = (
                self._apply_matrix_envelope_mode(
                    name=name,
                    pred_matrix=raw_pred_matrix,
                    target_matrix=raw_target_matrix,
                    x=x,
                )
            )
            matrix_edge_weights = (
                edge_loss_weights if name in {"hamiltonian", "overlap"} else None
            )
            need_irrep_cache = allow_train_metrics and (
                self.cfg.log_per_irrep_metrics
                or (
                    name == "hamiltonian"
                    and (
                        self.cfg.log_hamiltonian_irrep_contrib_metrics
                        or self.cfg.log_hamiltonian_pair_contrib_metrics
                    )
                )
            )
            need_hamiltonian_contribs = allow_train_metrics and (
                name == "hamiltonian"
                and (
                    self.cfg.log_hamiltonian_irrep_contrib_metrics
                    or self.cfg.log_hamiltonian_pair_contrib_metrics
                )
            )

            if self.cfg.train_on_irrep_parts:
                if self._matrix_envelope_mode() != "off":
                    raise ValueError(
                        "hamiltonian_envelope_mode is not supported with train_on_irrep_parts."
                    )
                target_irreps = raw_target_matrix.to_vectors(self.mapper)
                loss_mse_val, loss_mae_val, per_irrep_metrics = (
                    self._compute_irrep_part_losses(preds_irreps[name], target_irreps)
                )
                metric_mse_val = loss_mse_val
                metric_mae_val = loss_mae_val
            else:
                p = loss_pred_matrix
                t = loss_target_matrix

                should_symmetrize = self.cfg.symmetrize_output or stage == "val"
                if should_symmetrize:
                    p = (p + p.transpose()) * 0.5
                    if self._matrix_envelope_mode() == "normalize_target":
                        t = (t + t.transpose()) * 0.5

                if should_symmetrize:
                    metric_pred_matrix = (
                        metric_pred_matrix + metric_pred_matrix.transpose()
                    ) * 0.5

                p_items, t_items = p.pair_blocks, t.pair_blocks

                loss_mse_val = torch.tensor(0.0, device=self.device)
                loss_mae_val = torch.tensor(0.0, device=self.device)

                # Vectorized loss calculation
                for key in t_items.keys():
                    if key not in p_items.keys():
                        raise ValueError(f"Key {key} not found in predicted items.")
                    preds = p_items[key]
                    targets = t_items[key]

                    target_n = targets.shape[0]
                    if target_n <= 0:
                        continue
                    if preds.shape[0] < target_n:
                        raise ValueError(
                            f"Predicted blocks for matrix {name}, key {key} are too short: "
                            f"pred_len={preds.shape[0]} target_len={target_n}"
                        )
                    preds = preds[:target_n]
                    targets = targets[:target_n]
                    pred_edges = p.pair_edges[key][:, :target_n]
                    target_edges = t.pair_edges[key][:, :target_n]

                    if self.cfg.safety_checks:
                        assert torch.equal(
                            pred_edges, target_edges
                        ), f"Edge mismatch in block loss for matrix {name}, key {key}."

                    partial_mask = self._partial_train_mask(target_edges)
                    preds = preds[partial_mask]
                    targets = targets[partial_mask]
                    if preds.shape[0] == 0:
                        continue

                    edge_weights_selected = None
                    if matrix_edge_weights is not None:
                        selected_global_idx = x["edge_partitions"][key]["global_idx"][
                            :target_n
                        ][partial_mask]
                        edge_weights_selected = matrix_edge_weights.index_select(
                            0, selected_global_idx
                        )
                    weighted_pair_mse, weighted_pair_mae = (
                        self._weighted_edge_block_losses(
                            preds=preds,
                            targets=targets,
                            edge_weights=edge_weights_selected,
                        )
                    )
                    loss_mse_val += weighted_pair_mse
                    loss_mae_val += weighted_pair_mae
                    pair_losses[key] = (
                        (1 - self.cfg.loss_l1_fraction) * weighted_pair_mse
                        + self.cfg.loss_l1_fraction * weighted_pair_mae
                    )

                metric_mse_val, metric_mae_val = _compute_physical_matrix_metrics(
                    metric_pred_matrix,
                    raw_target_matrix,
                    name,
                )

            matrix_mses[name] = metric_mse_val
            matrix_maes[name] = metric_mae_val
            combined_pair_losses[name] = pair_losses

            if need_irrep_cache:
                pred_irrep_blocks = build_irrep_block_matrix_cache(
                    metric_pred_matrix, self.mapper, self.all_irreps
                )
                target_irrep_blocks = build_irrep_block_matrix_cache(
                    raw_target_matrix, self.mapper, self.all_irreps
                )
                irrep_block_cache_by_name[name] = (
                    pred_irrep_blocks,
                    target_irrep_blocks,
                )

            if need_hamiltonian_contribs:
                hamiltonian_mae_contribs = compute_hamiltonian_mae_contributions(
                    metric_pred_matrix,
                    raw_target_matrix,
                    self.mapper,
                    all_irreps=self.all_irreps,
                    compute_irrep_sums=self.cfg.log_hamiltonian_irrep_contrib_metrics,
                    compute_pair_sums=self.cfg.log_hamiltonian_pair_contrib_metrics,
                    require_exact_prefix=bool(self.cfg.require_exact_edge_match),
                    pred_irrep_blocks=pred_irrep_blocks,
                    target_irrep_blocks=target_irrep_blocks,
                )

            # Store for combined loss BEFORE unit conversion
            mse_for_loss = loss_mse_val
            mae_for_loss = loss_mae_val

            mse_val = metric_mse_val
            mae_val = metric_mae_val
            if name == "hamiltonian":
                # Convert to eV^2 and eV for logging only
                mse_val = metric_mse_val * (HARTREE_TO_EV**2)
                if hamiltonian_mae_contribs is not None:
                    denom = int(hamiltonian_mae_contribs["total_count"])
                    if denom > 0:
                        mae_val = (
                            float(hamiltonian_mae_contribs["total_abs_sum"]) / denom
                        ) * HARTREE_TO_EV
                else:
                    mae_val = metric_mae_val * HARTREE_TO_EV
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
        if self._stop_training_on_nonfinite_loss(
            loss_matrix,
            stage=stage,
            batch_idx=batch_idx,
            phase="matrix",
        ):
            return None

        # --- Observable Evaluation --------------------------------------
        t_obs_start = time.perf_counter()
        loss_E_weighted = torch.tensor(0.0, device=self.device)
        loss_N_weighted = torch.tensor(0.0, device=self.device)
        loss_F_weighted = torch.tensor(0.0, device=self.device)
        loss_spectral_weighted = torch.tensor(0.0, device=self.device)
        observable_preds_matrix = {}
        observable_trace_alignment = x["pred_trace_alignment"]

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
            H_true = y["hamiltonian"]
            D_true = y["density"]
            S_true = y["overlap"]

        E_true = y.get("energy")
        N_true = y.get("num_electrons")
        for name, pred_matrix in preds_matrix.items():
            target_matrix = y.get(name)
            if target_matrix is None:
                continue
            physical_pred_matrix = pred_matrix
            if name in {"hamiltonian", "overlap"}:
                mode = self._matrix_envelope_mode()
                if mode in {"multiply_prediction", "normalize_target"}:
                    physical_pred_matrix = scale_block_matrix_by_edge_values(
                        pred_matrix,
                        x["edge_envelope"],
                        x["edge_partitions"],
                    )
            observable_preds_matrix[name] = truncate_pred_block_matrix_to_target_prefix(
                physical_pred_matrix,
                target_matrix,
            )
        observable_values = build_observable_predictions(
            observable_preds_matrix,
            trace_alignment=observable_trace_alignment,
            H_true=H_true,
            D_true=D_true,
            S_true=S_true,
        )
        add_observable_metrics(
            metrics,
            stage=stage,
            cfg=self.cfg,
            observable_values=observable_values,
            energy_target=E_true,
            num_electrons_target=N_true,
        )
        loss_E_weighted, loss_N_weighted = observable_loss(
            cfg=self.cfg,
            observable_values=observable_values,
            energy_target=E_true,
            num_electrons_target=N_true,
            mse_fn=self._mse,
            device=self.device,
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

        if allow_train_metrics and self.cfg.log_per_irrep_metrics:
            for name in self.cfg.matrix_targets:
                if name not in preds_irreps or name not in y:
                    continue
                cache = irrep_block_cache_by_name.get(name)
                if cache is None:
                    cache = (
                        build_irrep_block_matrix_cache(
                            preds_matrix[name], self.mapper, self.all_irreps
                        ),
                        build_irrep_block_matrix_cache(
                            y[name], self.mapper, self.all_irreps
                        ),
                    )
                    irrep_block_cache_by_name[name] = cache
                pred_irrep_blocks, target_irrep_blocks = cache
                irrep_metrics = compute_irrep_metrics(
                    preds_matrix[name],
                    y[name],
                    self.all_irreps,
                    self.mapper,
                    pred_irrep_blocks=pred_irrep_blocks,
                    target_irrep_blocks=target_irrep_blocks,
                )
                for key, value in irrep_metrics.items():
                    metrics[f"{stage}/{name}_irrep_{key}"] = value

        t_obs_end = time.perf_counter()

        if (
            self._spectral_loss_enabled()
            and "hamiltonian" in observable_preds_matrix
            and float(getattr(self.cfg, "spectral_loss_coef", 0.0)) != 0.0
        ):
            spectral_loss, spectral_stats = self._compute_spectral_loss(
                x=x,
                physical_pred_hamiltonian=observable_preds_matrix["hamiltonian"],
            )
            loss_spectral_weighted = float(self.cfg.spectral_loss_coef) * spectral_loss
            metrics[f"{stage}/loss_spectral"] = spectral_loss
            metrics[f"{stage}/loss_spectral_weighted"] = loss_spectral_weighted
            metrics[f"{stage}/spectral_mae_ev"] = spectral_stats["spectral_mae_ev"]
            metrics[f"{stage}/spectral_weight_sum"] = spectral_stats[
                "spectral_weight_sum"
            ]

        # --- Total Loss Aggregation ---
        loss = (
            loss_matrix
            + loss_E_weighted
            + loss_N_weighted
            + loss_F_weighted
            + loss_spectral_weighted
        )

        # L1 and L2 regularization
        if self.cfg.l1_reg_coef > 0:
            l1_reg = sum(p.abs().sum() for p in self.parameters())
            loss += self.cfg.l1_reg_coef * l1_reg
            metrics[f"{stage}/loss_l1_reg"] = l1_reg
        if self.cfg.l2_reg_coef > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            loss += self.cfg.l2_reg_coef * l2_reg
            metrics[f"{stage}/loss_l2_reg"] = l2_reg

        if self._stop_training_on_nonfinite_loss(
            loss,
            stage=stage,
            batch_idx=batch_idx,
            phase="total",
        ):
            return None

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
        if loss_spectral_weighted > 0:
            metrics[f"{stage}/loss_spectral_total"] = loss_spectral_weighted

        for name, loss_val in combined_matrix_losses.items():
            metrics[f"{stage}/loss_block_{name}"] = loss_val

        if self.cfg.log_per_pair_loss_metrics:
            for matrix_name, pair_losses in combined_pair_losses.items():
                for pair_key, pair_loss in pair_losses.items():
                    metrics[f"{stage}/loss_block_{matrix_name}_{pair_key}"] = pair_loss

        if (
            hamiltonian_mae_contribs is not None
            and int(hamiltonian_mae_contribs["total_count"]) > 0
        ):
            denom = int(hamiltonian_mae_contribs["total_count"])
            if self.cfg.log_hamiltonian_irrep_contrib_metrics:
                for irrep_key, abs_sum in hamiltonian_mae_contribs[
                    "irrep_abs_sums"
                ].items():
                    metrics[f"{stage}/hamiltonian_mae_{irrep_key}"] = (
                        abs_sum / denom
                    ) * HARTREE_TO_EV
            if self.cfg.log_hamiltonian_pair_contrib_metrics:
                for pair_key, abs_sum in hamiltonian_mae_contribs[
                    "pair_abs_sums"
                ].items():
                    metrics[f"{stage}/hamiltonian_mae_{pair_key.replace('-', '_')}"] = (
                        abs_sum / denom
                    ) * HARTREE_TO_EV

        if stage == "train" and loss > 1e-12:
            for name, val in combined_matrix_losses.items():
                metrics[f"frac/loss_{name}"] = val / loss
            if self.cfg.log_per_pair_loss_metrics:
                for matrix_name, pair_losses in combined_pair_losses.items():
                    for pair_key, pair_loss in pair_losses.items():
                        metrics[f"frac/loss_{matrix_name}_{pair_key}"] = (
                            pair_loss / loss
                        )
            if loss_E_weighted > 0:
                metrics["frac/loss_energy"] = loss_E_weighted / loss
            if loss_N_weighted > 0:
                metrics["frac/loss_num_electrons"] = loss_N_weighted / loss
            if loss_F_weighted > 0:
                metrics["frac/loss_forces"] = loss_F_weighted / loss
            if loss_spectral_weighted > 0:
                metrics["frac/loss_spectral"] = loss_spectral_weighted / loss

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
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters remain after freeze settings.")
        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.lr)
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
        x: Dict[str, Any] | None = None,
    ) -> "Snapshot":
        if x is None and self._matrix_envelope_mode() != "off":
            raise ValueError(
                "predictions_to_snapshot requires the input batch `x` when "
                "hamiltonian_envelope_mode is enabled."
            )
        block_matrices = self.predicted_irreps_to_block_matrices(
            predictions,
            x if x is not None else {},
            physical=True,
        )
        return Snapshot(
            hamiltonian=(
                block_matrices["hamiltonian"]
                if "hamiltonian" in block_matrices
                else None
            ),
            overlap=(
                block_matrices["overlap"] if "overlap" in block_matrices else None
            ),
            density=(
                block_matrices["density"] if "density" in block_matrices else None
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
