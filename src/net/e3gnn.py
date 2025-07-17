"""
E3GNN – PyTorch-Lightning implementation
----------------------------------------

*  encoders.py         → NodeEncoder / EdgeEncoder
*  layers.py           → MessageBlock (small & large graphs)
*  heads.py            → DeepHead (H, S, D)
*  sparse_math.trace_* → Energy / electron count losses
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any

import torch
import pytorch_lightning as pl
from torch import nn
from omegaconf import DictConfig

from collections import OrderedDict

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.snapshot import Snapshot
from data.block_matrix import IrrepsBlockData

from net.common import HyperParams, build_hidden_irreps
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
        edge_types: List[str],
        cfg: DictConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # checkpoint
        self.save_hyperparameters(ignore=["mapper", "cfg"])
        self.hp = HyperParams(**cfg.model)
        self.lr = cfg.training.lr
        self.pedantic = cfg.logging.pedantic
        self.device_ = torch.device(device)
        # shared mapper
        self.mapper: BlockIrrepMapper = mapper.to(self.device_)

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = build_hidden_irreps(
            self.hp.l_max, self.hp.hidden_base_dim
        )
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.hp.l_max)

        # ---------- encoders -------------------------------------------
        self.node_enc = NodeEncoder(
            node_one_hot_dim=len(self.mapper.orbital_cfg.elements()),
            out_irreps=self.hidden_irreps,
            hp=self.hp,
            device=self.device_,
            dtype=dtype,
        )
        self.edge_enc = EdgeEncoder(
            n_edge_types=len(edge_types),
            n_radial=self.hp.n_radial,
            sh_irreps=self.sh_irreps,
            out_irreps=self.hidden_irreps,
            hp=self.hp,
            device=self.device_,
            dtype=dtype,
        )

        # ---------- message-passing stacks -----------------------------
        def _make_mp():
            return MessageBlock(
                self.hidden_irreps,
                self.hp,
                device=self.device_,
                dtype=dtype,
            )

        self.mp_small = nn.ModuleList(
            [_make_mp() for _ in range(self.hp.num_layers_gnn)]
        )
        self.mp_large = nn.ModuleList(
            [_make_mp() for _ in range(self.hp.num_layers_matrix)]
        )

        # ---------- heads ----------------------------------------------
        self.heads = nn.ModuleDict(
            {
                name: DeepHead(
                    in_irreps=self.hidden_irreps,
                    pair_keys=edge_types,
                    mapper=self.mapper,
                    hp=self.hp,
                    device=self.device_,
                    dtype=dtype,
                )
                for name in ("hamiltonian", "overlap", "density")
            }
        )

    # ------------------------------------------------------------------ util helpers
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

    # ------------------------------------------------------------------ activation monitoring helpers
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
        if self.pedantic:
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
        if self.pedantic:
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
            + self.hp.energy_loss_coef * loss_E
            + self.hp.electron_loss_coef * loss_N
        )
        # L1 and L2 regularization
        if self.hp.l1_reg_coef > 0:
            l1_reg = sum(p.abs().sum() for p in self.parameters())
            loss += self.hp.l1_reg_coef * l1_reg
        if self.hp.l2_reg_coef > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            loss += self.hp.l2_reg_coef * l2_reg

        # log all metrics
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_loss_blocks": loss_blocks,
            f"{stage}_loss_E": loss_E,
            f"{stage}_loss_N": loss_N,
            f"{stage}_abs_error_E": abs_err_E,
            f"{stage}_abs_error_N": abs_err_N,
        }
        if self.hp.l1_reg_coef > 0:
            metrics[f"{stage}_l1_reg"] = l1_reg
        if self.hp.l2_reg_coef > 0:
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
        snapshot = self.predictions_to_snapshot(predictions, positions, box)
        energy = snapshot.get_energy()
        grad = torch.autograd.grad(
            energy,
            positions,
            create_graph=True,  # needed for second derivatives (eg. training on forces)
        )[0]
        return -grad

    def predict_forces(self, x: Dict[str, Any]) -> torch.Tensor:
        predictions = self(x)
        return self.get_forces(predictions, x["positions"], x["box"])
