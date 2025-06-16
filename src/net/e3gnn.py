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

from core.orbital_irrep_config import OrbitalIrrepConfig
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

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        orbital_cfg: OrbitalIrrepConfig,
        edge_types: List[str],
        hp: HyperParams = HyperParams(),
        *,
        lr: float = 3e-4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["orbital_cfg"])  # Lightning checkpointing
        self.hp = hp
        self.lr = lr
        self.device_ = torch.device(device)

        # ---------- shared irreps ---------------------------------------
        self.hidden_irreps: Irreps = build_hidden_irreps(hp.l_max, hp.hidden_base_dim)
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(hp.l_max)

        # ---------- encoders -------------------------------------------
        self.node_enc = NodeEncoder(
            node_one_hot_dim=len(orbital_cfg.elements()),
            out_irreps=self.hidden_irreps,
            hp=hp,
            device=self.device_,
            dtype=dtype,
        )
        self.edge_enc = EdgeEncoder(
            n_edge_types=len(edge_types),
            n_radial=hp.n_radial,
            sh_irreps=self.sh_irreps,
            offdiag_irrep_dim=None,  # no overlap_offdiag in this rev
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
                    orbital_cfg=orbital_cfg,
                    hp=hp,
                    device=self.device_,
                    dtype=dtype,
                )
                for name in ("hamiltonian", "overlap", "density")
            }
        )

        # tiny mapper for wrapping predictions back to IrrepsBlockData
        self.mapper = BlockIrrepMapper(orbital_cfg, diagonal=False, device=self.device_)

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
            mapper=self.mapper,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x_gnn: Dict[str, Any],
        x_matrix: Dict[str, Any],
        atoms: Tuple[str, ...],
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
        # re-encode the *new* edges only (helper from original skeleton)
        edge_big = self._lift_edge_features(
            edge,
            x_gnn,
            x_matrix,
        )
        ei_big = x_matrix["edge_index"]
        for blk in self.mp_large:
            node, edge_big = blk(node, edge_big, ei_big)

        # ---- heads -----------------------------------------------------
        edge_type_idx = x_matrix["edge_one_hot"].argmax(dim=-1)
        preds_raw = {
            name: head(edge_big, edge_type_idx, ei_big)
            for name, head in self.heads.items()
        }
        preds_wrapped = {
            name: self._wrap_head_output(raw, atoms) for name, raw in preds_raw.items()
        }
        return preds_wrapped

    # ==================== Lightning steps ====================================
    def training_step(self, batch, batch_idx):
        x_gnn, x_mat, y = batch  # relies on custom collate_fn!
        atoms = tuple(y["snapshot"].atoms)  # same for whole batch after collate

        preds = self(x_gnn, x_mat, atoms)

        # ---- block losses ---------------------------------------------
        loss_blocks = 0.0
        for name in ("hamiltonian", "overlap", "density"):
            p = preds[name].pair_vectors
            t = y[name].pair_vectors
            for key in p:
                loss_blocks = loss_blocks + self._vectors_mse(p[key], t[key])

        # ---- energy / electrons ---------------------------------------
        E_pred = trace_matmul_sparse_snap_vectorized(
            preds["hamiltonian"].to_blocks(), preds["density"].to_blocks()
        )
        E_true = y["energy"]
        loss_E = torch.mean((E_pred - E_true) ** 2)

        N_pred = trace_matmul_sparse_snap_vectorized(
            preds["overlap"].to_blocks(), preds["density"].to_blocks()
        )
        N_true = y["num_electrons"]
        loss_N = torch.mean((N_pred - N_true) ** 2)

        loss = (
            loss_blocks
            + self.hp.energy_loss_coef * loss_E
            + self.hp.electron_loss_coef * loss_N
        )

        self.log_dict(
            {
                "loss": loss,
                "loss_blocks": loss_blocks,
                "loss_E": loss_E,
                "loss_N": loss_N,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)  # logs are identical

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
