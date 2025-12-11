import sys
from pathlib import Path

# Add project root to path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import yaml
import torch
import matplotlib.pyplot as plt

# Add DeepH-E3 path to allow direct imports
sys.path.append(str(project_root / "external/DeepH-E3"))

# Mandala imports
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper

# DeepH-E3 imports
from deephe3.e3modules import Rotate, e3TensorDecomp


def setup_study_dir(study_dir):
    study_dir.mkdir(parents=True, exist_ok=True)


def get_silicon_snapshot():
    """Loads the silicon snapshot using Mandala's data pipeline."""
    # This is a simplified version of the DatasetFactory logic
    from data.openmx_parser import parse_openmx_scfout
    from data.openmx_info_parser import parse_info_out

    matrix_path = project_root / "data/big/silicon/900K/Si_DM"
    info_path = project_root / "data/big/silicon/900K/info.txt"

    info = parse_info_out(info_path)
    atoms = info.elements
    orb_cfg = OrbitalIrrepConfig.from_dict(info.orbital_set)
    print(orb_cfg)

    # Mandala way: convert basis on load
    snap_mandala = parse_openmx_scfout(matrix_path, atoms, orb_cfg, convention="e3nn")

    # DeepH-E3 way: load as-is
    snap_deeph_raw = parse_openmx_scfout(
        matrix_path, atoms, orb_cfg, convention="openmx"
    )

    return snap_mandala, snap_deeph_raw, orb_cfg


def compare_vectors(v_mandala, v_deeph, report, plot_path):
    abs_diff = torch.abs(v_mandala - v_deeph)
    rel_diff = abs_diff / (torch.abs(v_mandala) + 1e-8)
    report["max_abs_diff"] = abs_diff.max().item()
    report["mean_abs_diff"] = abs_diff.mean().item()
    report["max_rel_diff"] = rel_diff.max().item()
    report["mean_rel_diff"] = rel_diff.mean().item()

    plt.figure(figsize=(6, 6))
    plt.scatter(v_mandala.detach().numpy(), v_deeph.detach().numpy(), alpha=0.5)
    plt.plot(
        [v_mandala.min(), v_mandala.max()], [v_mandala.min(), v_mandala.max()], "r--"
    )
    plt.xlabel("Mandala")
    plt.ylabel("DeepH-E3")
    plt.title("Comparison Plot")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def experiment_1_basis_equivalence(snap_mandala, snap_deeph_raw, orb_cfg, study_dir):
    print("Running Experiment 1: Basis Transformation Equivalence")
    report = {}

    # --- Part 1a: Training on Irreps ---
    print("  Part 1a: Comparing irrep vectors...")
    report["irreps_target"] = {}

    mapper_mandala = BlockIrrepMapper(orb_cfg)
    pair_key = "Si-Si"
    block_mandala_e3nn = snap_mandala.hamiltonian[pair_key][0]
    vector_mandala = mapper_mandala.blocks_to_vectors(pair_key, block_mandala_e3nn)

    # --- DeepH-E3 Path ---
    # 1. Get the raw OpenMX block
    block_deeph_openmx = snap_deeph_raw.hamiltonian[pair_key][0]

    # 2. Instantiate e3TensorDecomp to get the conversion logic
    # We need to provide it with the expected output irreps for the Si-Si block.
    # This is equivalent to what BlockIrrepMapper does internally.
    si_irreps = orb_cfg.element_to_irreps["Si"]
    out_js_list = [
        mul * [(ir1.l, ir2.l)] for mul, ir1 in si_irreps for _, ir2 in si_irreps
    ]
    out_js_list = [ir_pair for sublist in out_js_list for ir_pair in sublist]  # Flatten
    # For this test, we only care about the Si-Si block, so we can simplify.
    # The actual out_js_list would come from the DeepH-E3 config.
    # Let's assume a simplified target block for this test.

    # We need a dummy net_irreps_out, which should match the required irreps.
    # Let e3TensorDecomp calculate the required irreps for us.
    decomp_helper = e3TensorDecomp(
        net_irreps_out=None, out_js_list=out_js_list, default_dtype_torch=torch.float32
    )
    net_irreps_out_required = decomp_helper.required_irreps_out

    # Now instantiate the real one for the conversion
    tensor_decomp = e3TensorDecomp(
        net_irreps_out=net_irreps_out_required,
        out_js_list=out_js_list,
        default_dtype_torch=torch.float32,
    )

    # 3. Use get_net_out to convert the block to an irrep vector
    # The method expects a batch dimension and a flattened H, so we add them.
    vector_deeph = tensor_decomp.get_net_out(block_deeph_openmx.flatten().unsqueeze(0))

    compare_vectors(
        vector_mandala.flatten(),
        vector_deeph.flatten(),
        report["irreps_target"],
        study_dir / "basis_irreps.png",
    )

    # --- Part 1b: Training on Matrix Elements ---
    print("  Part 1b: Comparing matrix elements...")
    report["matrix_target"] = {}
    target_matrix_mandala = snap_mandala.hamiltonian[pair_key][0]

    # DeepH-E3 converts on the fly. We replicate this.
    rotator_deeph = Rotate(torch.float32)
    # We need the orbital types to do the full block conversion
    si_orbital_types = [mul * [ir.l] for mul, ir in orb_cfg.element_to_irreps["Si"]]
    target_matrix_deeph_e3nn = rotator_deeph.openmx2wiki_H_full(
        block_deeph_openmx, si_orbital_types, si_orbital_types
    )

    compare_vectors(
        target_matrix_mandala.flatten(),
        target_matrix_deeph_e3nn.flatten(),
        report["matrix_target"],
        study_dir / "basis_matrix.png",
    )

    with open(study_dir / "basis_equivalence_report.yaml", "w") as f:
        yaml.dump(report, f)

    print("Experiment 1 finished. Report saved.")


def main():
    study_dir = Path("studies/deeph-e3_comparison")
    print(f"Creating study in {study_dir.resolve()}")
    setup_study_dir(study_dir)

    print("Loading data...")
    snap_mandala, snap_deeph_raw, orb_cfg = get_silicon_snapshot()

    experiment_1_basis_equivalence(snap_mandala, snap_deeph_raw, orb_cfg, study_dir)

    print("Study finished.")


if __name__ == "__main__":
    main()
