from pathlib import Path
import sys
import torch
from ase.io import read as ase_read
from e3nn.o3 import Irreps

sys.path.append("src")
from data.openmx_info_parser import parse_info_out, recover_box
from data.graph_features import compute_graph_features
from net.common import Config

base = Path("data/small/ZnCu2Sn_SeS_2_scale_1_010")
info = parse_info_out(base / "ZnCuSeS.out", dtype=torch.float64)
cif = ase_read(base / "ZnCuSeS.cif")

atoms = tuple(cif.get_chemical_symbols())
pos_gt = torch.tensor(cif.get_positions(), dtype=torch.float64)
box_A = torch.tensor(cif.cell.array, dtype=torch.float64)
pos_out = info.positions.to(dtype=torch.float64)
box_B = recover_box(info.frac.to(dtype=torch.float64), pos_out).to(dtype=torch.float64)

invA = torch.linalg.inv(box_A)
delta_frac = (pos_out - pos_gt) @ invA
atom_shifts = torch.round(delta_frac).to(torch.long)
resid = delta_frac - atom_shifts
print("max atom-image residual vs CIF box:", float(resid.abs().max().item()))
print(
    "unique atom shifts:",
    {
        tuple(int(v) for v in row.tolist()): int(
            (atom_shifts == row).all(dim=1).sum().item()
        )
        for row in atom_shifts.unique(dim=0)
    },
)

cfg = Config(cutoff_radius=11.0, n_radial=8, safety_checks=True)
sh_irreps = Irreps("1x0e + 1x1o")
edge_type2idx = {
    f"{a}-{b}": i
    for i, (a, b) in enumerate(
        [
            (x, y)
            for x in ["Zn", "Cu", "Sn", "Se", "S"]
            for y in ["Zn", "Cu", "Sn", "Se", "S"]
        ]
    )
}
# use the actual edge types known by the mapper would be better, but for a one-off geometry check
# the key comparison we care about is using the same atom set; build only the keys that exist.
# We'll derive the set from the snapshots below.

# build a minimal edge_type map from observed atom pairs in the CIF geometry
observed = sorted(
    {f"{atoms[i]}-{atoms[j]}" for i in range(len(atoms)) for j in range(len(atoms))}
)
edge_type2idx = {k: i for i, k in enumerate(observed)}

out_ref = compute_graph_features(pos_gt, box_A, atoms, cfg, sh_irreps, edge_type2idx)
out_shift = compute_graph_features(pos_out, box_B, atoms, cfg, sh_irreps, edge_type2idx)

edge_index_A, edge_shift_A, edge_type_idx_A, edge_length_emb_A, edge_sh_A, n_self_A = (
    out_ref
)
edge_index_B, edge_shift_B, edge_type_idx_B, edge_length_emb_B, edge_sh_B, n_self_B = (
    out_shift
)

print("edge_index equal:", torch.equal(edge_index_A, edge_index_B))
print("edge_type_idx equal:", torch.equal(edge_type_idx_A, edge_type_idx_B))
print("edge_length_emb equal:", torch.allclose(edge_length_emb_A, edge_length_emb_B))
print("edge_sh equal:", torch.allclose(edge_sh_A, edge_sh_B))
print("num_self_edges equal:", n_self_A == n_self_B)

# Compare edge shifts after relabeling one position set to the other convention.
# Compute the per-atom lattice shift from pos_out to pos_gt.
atom_shifts = torch.round((pos_out - pos_gt) @ invA).to(torch.long)
src = edge_index_A[0]
dst = edge_index_A[1]
expected_edge_shift_B = edge_shift_A + atom_shifts[src].T - atom_shifts[dst].T
print("edge_shift relabel match:", torch.equal(edge_shift_B, expected_edge_shift_B))
print(
    "max abs edge_shift diff after relabel:",
    int((edge_shift_B - expected_edge_shift_B).abs().max().item()),
)
