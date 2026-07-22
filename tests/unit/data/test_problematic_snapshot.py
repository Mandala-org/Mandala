from pathlib import Path

import pytest
import torch
from ase.io import read as ase_read
from e3nn.o3 import Irreps

from data.openmx_info_parser import parse_info_out, recover_box
from data.graph_features import compute_graph_features
from net.common import Config


@pytest.mark.integration
def test_graph_features_are_invariant_for_historical_atom_image_relabeling_case():
    """Preserve a real regression where one atom is represented in another image."""
    base = Path("data/small/ZnCu2Sn_SeS_2_scale_1_010")
    info = parse_info_out(base / "ZnCuSeS.out", dtype=torch.float64)
    cif = ase_read(base / "ZnCuSeS.cif")

    atoms = tuple(cif.get_chemical_symbols())
    pos_gt = torch.tensor(cif.get_positions(), dtype=torch.float64)
    box_ref = torch.tensor(cif.cell.array, dtype=torch.float64)
    pos_out = info.positions.to(dtype=torch.float64)
    box_recovered = recover_box(info.frac.to(dtype=torch.float64), pos_out).to(
        dtype=torch.float64
    )

    inv_box = torch.linalg.inv(box_ref)
    delta_frac = (pos_out - pos_gt) @ inv_box
    atom_shifts = torch.round(delta_frac).to(torch.long)
    residual = delta_frac - atom_shifts
    assert residual.abs().max() < 1e-5
    assert torch.any(atom_shifts != 0), "Regression fixture must include an image shift"

    cfg = Config(cutoff_radius=11.0, n_radial=8, safety_checks=True)
    sh_irreps = Irreps("1x0e + 1x1o")
    observed = sorted({f"{src}-{dst}" for src in atoms for dst in atoms})
    edge_type2idx = {key: idx for idx, key in enumerate(observed)}

    out_ref = compute_graph_features(
        pos_gt, box_ref, atoms, cfg, sh_irreps, edge_type2idx
    )
    out_shifted = compute_graph_features(
        pos_out, box_recovered, atoms, cfg, sh_irreps, edge_type2idx
    )

    (
        edge_index_ref,
        edge_shift_ref,
        edge_type_idx_ref,
        edge_length_emb_ref,
        edge_sh_ref,
        n_self_ref,
        edge_lengths_ref,
    ) = out_ref
    (
        edge_index_shifted,
        edge_shift_shifted,
        edge_type_idx_shifted,
        edge_length_emb_shifted,
        edge_sh_shifted,
        n_self_shifted,
        edge_lengths_shifted,
    ) = out_shifted

    assert n_self_ref == n_self_shifted

    # The recovered cell differs at about 1e-4 Angstrom, so near-tied edges may
    # sort differently. Match the physical edges instead of their column order.
    shifted_in_ref_convention = (
        edge_shift_shifted
        - atom_shifts[edge_index_shifted[0]].T
        + atom_shifts[edge_index_shifted[1]].T
    )

    def edge_keys(edge_index, edge_shift):
        return [
            (
                int(edge_index[0, idx]),
                int(edge_index[1, idx]),
                *(int(value) for value in edge_shift[:, idx]),
            )
            for idx in range(edge_index.shape[1])
        ]

    ref_keys = edge_keys(edge_index_ref, edge_shift_ref)
    shifted_keys = edge_keys(edge_index_shifted, shifted_in_ref_convention)
    assert len(ref_keys) == len(set(ref_keys))
    assert set(ref_keys) == set(shifted_keys)

    shifted_index = {key: idx for idx, key in enumerate(shifted_keys)}
    reorder = torch.tensor([shifted_index[key] for key in ref_keys])
    assert torch.equal(edge_type_idx_ref, edge_type_idx_shifted[reorder])
    assert torch.allclose(edge_lengths_ref, edge_lengths_shifted[reorder], atol=2e-4)
    assert torch.allclose(
        edge_length_emb_ref, edge_length_emb_shifted[reorder], atol=1e-4
    )
    assert torch.allclose(edge_sh_ref, edge_sh_shifted[reorder], atol=1e-4)
