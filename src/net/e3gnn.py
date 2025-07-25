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

from collections import OrderedDict

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.snapshot import Snapshot
from data.block_matrix import IrrepsBlockData

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
        self.device = torch.device(cfg.device)
        # shared mapper
        self.mapper: BlockIrrepMapper = mapper.to(self.device)

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
            self.cfg.l_max, self.cfg.hidden_base_dim
        )
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max)

        # ---------- encoders -------------------------------------------
        self.node_enc = NodeEncoder(
            node_one_hot_dim=len(self.mapper.orbital_cfg.elements()),
            out_irreps=self.hidden_irreps,
            cfg=self.cfg,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.edge_enc = EdgeEncoder(
            n_edge_types=len(self.mapper._maps.keys()),
            n_radial=self.cfg.n_radial,
            sh_irreps=self.sh_irreps,
            out_irreps=self.hidden_irreps,
            cfg=self.cfg,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        # ---------- message-passing stacks -----------------------------
        def _make_mp():
            return MessageBlock(
                self.hidden_irreps,
                self.cfg,
                device=self.device,
                dtype=self.cfg.dtype,
            )

        self.mp_small = nn.ModuleList(
            [_make_mp() for _ in range(self.cfg.num_layers_gnn)]
        )
        self.mp_large = nn.ModuleList(
            [_make_mp() for _ in range(self.cfg.num_layers_matrix)]
        )

        # ---------- heads ----------------------------------------------
        pair_keys = list(
            map(lambda pair: f"{pair[0]}-{pair[1]}", self.mapper._maps.keys())
        )
        self.heads = nn.ModuleDict(
            {
                name: DeepHead(
                    in_irreps=self.hidden_irreps,
                    pair_keys=pair_keys,
                    mapper=self.mapper,
                    cfg=self.cfg,
                    device=self.device,
                    dtype=self.cfg.dtype,
                )
                for name in ("hamiltonian", "overlap", "density")
            }
        )

    # ------------------------ util helpers -----------------------------
    @staticmethod
    def _vectors_mse(pred, target):
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch in block loss")
        return torch.mean((pred - target) ** 2)

    def _wrap_head_output(
        self,
        raw: Dict[str, Dict[str, torch.Tensor]],
        atoms: Tuple[str, ...],
    ):
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
            for idx, (i, j) in enumerate(edges.t().tolist()):
                lookup[(i, j)] = (key, idx)

        from data.block_matrix import IrrepsBlockData  # local import

        return IrrepsBlockData(
            atoms=atoms,
            atom_counts=Counter(atoms),
            pair_vectors=pair_vec,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=self.mapper.orbital_cfg,
        )

    # --------------- activation monitoring helpers ----------------------------
    def _magnitude_splits(
        self,
        features: torch.Tensor,
        irreps: Irreps,
    ) -> dict[str, torch.Tensor]:
        """
        Split features by irrep and compute magnitude per irreducible component.
        Returns a mapping from angular momentum l to a 1D tensor of magnitudes.
        """
        mags: dict[str, torch.Tensor] = OrderedDict()
        # features: (M, D)
        start = 0
        for mul, ir in irreps:
            dim = ir.dim
            size = mul * dim
            # slice for this irrep
            chunk = features[:, start : start + size]
            # reshape to (M * mul, dim)
            if mul > 0 and dim > 0:
                reshaped = chunk.reshape(-1, dim)
                # magnitude across dim
                mag = torch.linalg.norm(reshaped, dim=1)
                mags[f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"] = mag
            start += size
        return mags

    def _record_activation_mags(
        self,
        prefix: str,
        features: torch.Tensor,
        irreps: Irreps,
    ) -> None:
        """
        Compute activation magnitudes and store in self._activation_mags.
        """
        splits = self._magnitude_splits(features, irreps)
        for ir_str, mag in splits.items():
            tag = f"mag_{prefix}_{ir_str}"
            self._activation_mags[tag] = mag

    # ------------------------------------------------------------------ forward
    def forward(self, x: Dict[str, Any]):
        # initialize activation magnitudes storage
        self._activation_mags: dict[str, torch.Tensor] = OrderedDict()
        # ---- encode ----------------------------------------------------
        node = self.node_enc(x["node_type_idx"])
        # record node encoding magnitudes per irrep
        self._record_activation_mags("node_encoding", node, self.hidden_irreps)
        edge = self.edge_enc(
            x["edge_type_idx"],
            x["edge_length_emb"],
            x["edge_sh"],
        )
        if self.cfg.pedantic:
            assert edge.requires_grad, "Gradients not flowing through edge encoder!"
        # record edge encoding magnitudes per irrep
        self._record_activation_mags("edge_encoding", edge, self.hidden_irreps)

        # ---- message-passing -------------------------------------------
        index_gnn_cutoff = x["index_gnn_cutoff"]
        edge_small = edge[:index_gnn_cutoff]
        ei_small = x["edge_index"][:, :index_gnn_cutoff]

        for idx, blk in enumerate(self.mp_small):
            node, edge_small = blk(node, edge_small, ei_small)
            # record magnitudes after small graph layer
            self._record_activation_mags(
                f"node_small_layer_{idx}", node, self.hidden_irreps
            )
            self._record_activation_mags(
                f"edge_small_layer_{idx}", edge_small, self.hidden_irreps
            )

        edge_large = edge[index_gnn_cutoff:]
        edge = torch.cat([edge_small, edge_large], dim=0)

        # ---- heads -----------------------------------------------------
        preds_raw = {
            name: head(edge, x["edge_type_idx"], x["edge_index"])
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
        preds = self(x)
        t_fwd_end = time.perf_counter()
        # block losses
        loss_blocks = 0.0
        if self.cfg.pedantic:
            for name in ("hamiltonian", "overlap", "density"):
                pred_edges = preds[name].pair_edges
                target_edges = y[name].pair_edges
                for key in target_edges.keys():
                    assert torch.equal(
                        pred_edges[key], target_edges[key]
                    ), f"Pedantic check failed: Edge order mismatch in '{name}' matrix for key '{key}'"

        for name in ("hamiltonian", "overlap", "density"):
            p_vecs = preds[name].pair_vectors
            t_vecs = y[name].pair_vectors
            for key in p_vecs:
                loss_blocks = loss_blocks + self._vectors_mse(p_vecs[key], t_vecs[key])
        # --- block mapping timing ---------------------------------------
        t_map_start = time.perf_counter()
        blk_ham = preds["hamiltonian"].to_blocks(self.mapper)
        blk_den = preds["density"].to_blocks(self.mapper)
        blk_ovr = preds["overlap"].to_blocks(self.mapper)
        t_map_end = time.perf_counter()
        # --- observable evaluation timing --------------------------------
        t_obs_start = time.perf_counter()
        E_pred = trace_matmul_sparse_snap_vectorized(blk_ham, blk_den)
        N_pred = trace_matmul_sparse_snap_vectorized(blk_ovr, blk_den)
        t_obs_end = time.perf_counter()
        E_true = y["energy"]
        loss_E = torch.mean((E_pred - E_true) ** 2)
        abs_err_E = torch.mean(torch.abs(E_pred - E_true))
        # electron count loss and absolute error
        N_true = y["num_electrons"]
        loss_N = torch.mean((N_pred - N_true) ** 2)
        abs_err_N = torch.mean(torch.abs(N_pred - N_true))
        # total loss
        loss = (
            loss_blocks
            + self.cfg.loss_coef_energy * loss_E
            + self.cfg.loss_coef_num_electrons * loss_N
        )
        # L1 and L2 regularization
        if self.cfg.l1_reg_coef > 0:
            l1_reg = sum(p.abs().sum() for p in self.parameters())
            loss += self.cfg.l1_reg_coef * l1_reg
        if self.cfg.l2_reg_coef > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            loss += self.cfg.l2_reg_coef * l2_reg

        # log all metrics
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_loss_blocks": loss_blocks,
            f"{stage}_loss_E": loss_E,
            f"{stage}_loss_N": loss_N,
            f"{stage}_abs_error_E": abs_err_E,
            f"{stage}_abs_error_N": abs_err_N,
        }
        if self.cfg.l1_reg_coef > 0:
            metrics[f"{stage}_l1_reg"] = l1_reg
        if self.cfg.l2_reg_coef > 0:
            metrics[f"{stage}_l2_reg"] = l2_reg
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
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
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    # ------------------------------------------------------------------ force prediction
    def predictions_to_snapshot(
        self,
        predictions: Dict[str, IrrepsBlockData],
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> "Snapshot":
        return Snapshot(
            hamiltonian=predictions["hamiltonian"].to_blocks(self.mapper),
            overlap=predictions["overlap"].to_blocks(self.mapper),
            density=predictions["density"].to_blocks(self.mapper),
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
            create_graph=True,  # needed for second derivatives (eg. training on forces)
        )[0]
        return -grad_pos

    def get_stress(
        self,
        predictions: Dict[str, IrrepsBlockData],
        positions: torch.Tensor,
        box: torch.Tensor,
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
        snapshot = self.predictions_to_snapshot(predictions, positions, box)
        energy = snapshot.get_energy()
        # 1. get dE/dh
        grad_box = torch.autograd.grad(
            energy,
            box,
            create_graph=True,
        )
        # 2. compute volume
        volume = torch.det(box)
        # 3. form Cauchy stress: σ_{αβ} = (1/Ω) ∑_γ h_{γα} (dE/dh_{γβ})
        #    - no minus sign here, since σ =  +∂E/∂ε /Ω
        stress = torch.matmul(box.t(), grad_box) / volume
        # 4. optionally symmetrize: σ → (σ+σ^T)/2
        stress = 0.5 * (stress + stress.transpose(-1, -2))
        return stress

    def predict_forces(self, x: Dict[str, Any]) -> torch.Tensor:
        predictions = self(x)
        return self.get_forces(predictions, x["positions"], x["box"])

    def predict_stress(self, x: Dict[str, Any]) -> torch.Tensor:
        predictions = self(x)
        return self.get_stress(predictions, x["positions"], x["box"])
