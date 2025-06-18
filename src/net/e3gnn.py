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

from e3nn.o3 import Irreps
import time

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_snap_vectorized

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
        hp: HyperParams = HyperParams(),
        *,
        lr: float = 3e-4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # check-point everything *except* the (non-serialisable) mapper
        self.save_hyperparameters(ignore=["mapper"])
        self.hp = hp
        self.lr = lr
        self.device_ = torch.device(device)
        # shared mapper (no local creation!)
        self.mapper: BlockIrrepMapper = mapper.to(self.device_)

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = build_hidden_irreps(hp.l_max, hp.hidden_base_dim)
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(hp.l_max)

        # ---------- encoders -------------------------------------------
        self.node_enc = NodeEncoder(
            node_one_hot_dim=len(self.mapper.orbital_cfg.elements()),
            out_irreps=self.hidden_irreps,
            hp=hp,
            device=self.device_,
            dtype=dtype,
        )
        self.edge_enc = EdgeEncoder(
            n_edge_types=len(edge_types),
            n_radial=hp.n_radial,
            sh_irreps=self.sh_irreps,
            offdiag_irrep_dim=None,  # off-diagonal overlap features disabled
            out_irreps=self.hidden_irreps,
            hp=hp,
            device=self.device_,
            dtype=dtype,
        )

        # ---------- message-passing stacks -----------------------------
        def _make_mp():
            return MessageBlock(
                self.hidden_irreps,
                hp,
                device=self.device_,
                dtype=dtype,
            )

        self.mp_small = nn.ModuleList([_make_mp() for _ in range(hp.num_layers_gnn)])
        self.mp_large = nn.ModuleList([_make_mp() for _ in range(hp.num_layers_matrix)])

        # ---------- heads ----------------------------------------------
        self.heads = nn.ModuleDict(
            {
                name: DeepHead(
                    in_irreps=self.hidden_irreps,
                    pair_keys=edge_types,
                    mapper=self.mapper,
                    hp=hp,
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
        Convert DeepHead raw dict → IrrepsBlockData with shared mapper.
        """
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
            pair_vectors=pair_vec,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=self.mapper.orbital_cfg,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x_gnn: Dict[str, Any],
        x_matrix: Dict[str, Any],
    ):
        # ---- encode ----------------------------------------------------
        node = self.node_enc(x_gnn["node_one_hot"])
        edge = self.edge_enc(
            x_gnn["edge_one_hot"],
            x_gnn["edge_length_emb"],
            x_gnn["edge_sh"],
            overlap_off=None,
        )

        ei_small = x_gnn["edge_index"]
        for blk in self.mp_small:
            node, edge = blk(node, edge, ei_small)

        # ---- lift to large graph --------------------------------------
        # re-encode edges for the large graph
        edge_big = self._lift_edge_features(
            edge,
            x_gnn,
            x_matrix,
        )
        ei_big = x_matrix["edge_index"]
        for blk in self.mp_large:
            node, edge_big = blk(node, edge_big, ei_big)

        # ---- combine node & edge features for heads --------------------
        # node features correspond to diagonal blocks
        batch_device = node.device
        # number of atoms / nodes
        N = node.shape[0]
        # stack node (diagonal) and edge (off-diagonal) representations
        h_full = torch.cat([node, edge_big], dim=0)
        # build full edge_index: self-loops first, then original matrix edges
        diag_idx = torch.arange(N, device=batch_device)
        diag_edge_index = torch.stack([diag_idx, diag_idx], dim=0)
        full_edge_index = torch.cat([diag_edge_index, x_matrix["edge_index"]], dim=1)
        # compute edge type indices: off-diags from x_matrix, diag from node elements
        edge_type_idx = x_matrix["edge_one_hot"].argmax(dim=-1)
        node_elem_idx = x_gnn["node_one_hot"].argmax(dim=-1)
        # lookup pair_keys (shared across heads)
        pair_keys = next(iter(self.heads.values())).pair_keys
        # build mapping from element → diag pair index
        elems = self.mapper.orbital_cfg.elements()
        diag_pair_idx = torch.tensor(
            [pair_keys.index(f"{el}-{el}") for el in elems],
            dtype=torch.long,
            device=batch_device,
        )
        diag_type_idx = diag_pair_idx[node_elem_idx]
        # concatenate diag + off-diag type indices
        full_edge_type_idx = torch.cat([diag_type_idx, edge_type_idx], dim=0)

        # ---- heads -----------------------------------------------------
        preds_raw = {
            name: head(h_full, full_edge_type_idx, full_edge_index)
            for name, head in self.heads.items()
        }
        preds_wrapped = {
            name: self._wrap_head_output(raw, tuple(x_gnn["atoms"]))
            for name, raw in preds_raw.items()
        }
        return preds_wrapped

    # ==================== Lightning steps ====================================
    def _shared_step(self, batch, batch_idx, stage: str):
        """
        Shared logic for training and validation steps.
        Logs metrics prefixed with stage ('train' or 'val').
        """
        x_gnn, x_mat, y = batch
        # --- forward timing (message-passing + heads) -------------------
        t_fwd_start = time.perf_counter()
        preds = self(x_gnn, x_mat)
        t_fwd_end = time.perf_counter()
        # block losses
        loss_blocks = 0.0
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
        N_pred = trace_matmul_sparse_snap_vectorized(
            preds["overlap"].to_blocks(self.mapper),
            preds["density"].to_blocks(self.mapper),
        )
        N_true = y["num_electrons"]
        loss_N = torch.mean((N_pred - N_true) ** 2)
        abs_err_N = torch.mean(torch.abs(N_pred - N_true))
        # total loss
        loss = (
            loss_blocks
            + self.hp.energy_loss_coef * loss_E
            + self.hp.electron_loss_coef * loss_N
        )
        # log all metrics
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_loss_blocks": loss_blocks,
            f"{stage}_loss_E": loss_E,
            f"{stage}_loss_N": loss_N,
            f"{stage}_abs_error_E": abs_err_E,
            f"{stage}_abs_error_N": abs_err_N,
        }
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

    # ------------------------------------------------------------------ edge lifting helper
    def _lift_edge_features(self, edge_small, x_small, x_large):
        """
        Re-encode only those edges present in x_large but missing in x_small.
        """
        ei_small = x_small["edge_index"]
        ei_big = x_large["edge_index"]

        small_map = {
            tuple(ei_small[:, k].tolist()): k for k in range(ei_small.shape[1])
        }
        E_big = ei_big.shape[1]
        device = edge_small.device
        out = torch.zeros(E_big, edge_small.shape[-1], device=device)

        missing_idx = []
        for k in range(E_big):
            tup = tuple(ei_big[:, k].tolist())
            if tup in small_map:
                out[k] = edge_small[small_map[tup]]
            else:
                missing_idx.append(k)

        if missing_idx:
            k_idx = torch.tensor(missing_idx, device=device, dtype=torch.long)
            enc = self.edge_enc(
                x_large["edge_one_hot"][k_idx],
                x_large["edge_length_emb"][k_idx],
                x_large["edge_sh"][k_idx],
                overlap_off=None,
            )
            out[k_idx] = enc
        return out
