"""
Utilities for exploring different data transformation variants.

This module provides configurable transformations for the OpenMX → E3NN pipeline
to systematically test alternative implementations and identify potential bugs.
"""

import torch
from pathlib import Path
from typing import Optional, Dict

from data.snapshot import Snapshot
from data.openmx_info_parser import parse_info_out
from data.openmx_parser import parse_openmx_scfout
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix


class ConfigurableBasisConverter:
    """
    Like OpenMXE3NNConverter but with configurable transformation behavior.

    Allows testing different SH basis transformation variants to identify
    potential bugs in the standard implementation.
    """

    def __init__(
        self,
        orbital_cfg: OrbitalIrrepConfig,
        mode: str = "standard",
        l_overrides: Optional[Dict[int, str]] = None,
        device="cpu",
    ):
        """
        Args:
            orbital_cfg: Orbital configuration
            mode: Transformation mode - "standard", "inverse", "none"
            l_overrides: Override specific l with custom permutation
                        e.g., {1: "1,2,0"} for p orbitals
            device: torch device
        """
        self.cfg = orbital_cfg
        self.mode = mode
        self.l_overrides = l_overrides or {}
        self.device = torch.device(device)

        # Build U matrices based on mode
        self._build_matrices()

    def _build_matrices(self):
        """Build transformation matrices based on mode and overrides."""
        from core.basis_converter import _U_OPENMX_TO_WIKI, _U_WIKI_TO_OPENMX

        self._U_openmx2wiki = {}

        for el in self.cfg.elements():
            # Get orbital types for this element
            irreps = self.cfg.element_to_irreps[el]
            typs = []
            for mul, (l, _p) in irreps:
                typs.extend([l] * mul)

            mats = []
            for l in typs:
                # Check for override
                if l in self.l_overrides:
                    perm_str = self.l_overrides[l]
                    if perm_str == "none":
                        mat = torch.eye(2 * l + 1, dtype=torch.float32)
                    else:
                        perm = [int(x) for x in perm_str.split(",")]
                        mat = torch.eye(2 * l + 1, dtype=torch.float32)[perm]
                else:
                    # Use mode
                    if self.mode == "standard":
                        mat = _U_OPENMX_TO_WIKI[l]
                    elif self.mode == "inverse":
                        mat = _U_WIKI_TO_OPENMX[l]
                    elif self.mode == "none":
                        mat = torch.eye(2 * l + 1, dtype=torch.float32)
                    else:
                        raise ValueError(f"Unknown mode: {self.mode}")

                mats.append(mat)

            self._U_openmx2wiki[el] = torch.block_diag(*mats).to(self.device)

    def block_transform(self, key: str, block: torch.Tensor) -> torch.Tensor:
        """Apply transformation to a block matrix."""
        el_i, el_j = key.split("-")
        U_i = self._U_openmx2wiki[el_i]
        U_j = self._U_openmx2wiki[el_j]
        return U_i @ block @ U_j.T

    def matrix_transform(self, matrix: BlockMatrix) -> BlockMatrix:
        """Transform all blocks in a BlockMatrix."""
        new_blocks = {
            k: self.block_transform(k, blk) for k, blk in matrix.pair_blocks.items()
        }
        return matrix._replace_pair_blocks(new_blocks, basis="e3nn")


def apply_coordinate_transform(
    tensor: torch.Tensor,
    permutation: Optional[str] = "2,0,1",
    mirror_x: bool = False,
) -> torch.Tensor:
    """
    Apply coordinate transformation (permutation and/or mirroring).

    Args:
        tensor: Positions or box tensor (N, 3) or (3, 3)
        permutation: Permutation string like "2,0,1" or "none"
        mirror_x: Whether to mirror the x coordinate

    Returns:
        Transformed tensor
    """
    if tensor is None:
        return None

    result = tensor.clone()

    # Apply mirroring first (in original coordinate system)
    if mirror_x:
        result[..., 0] = -result[..., 0]

    # Apply permutation
    if permutation and permutation != "none":
        perm = [int(x) for x in permutation.split(",")]
        change_of_basis = torch.eye(3, dtype=torch.float32)[perm]
        result = result @ change_of_basis

    return result


def load_snapshot_with_transforms(
    matrix_path: str | Path,
    info_path: str | Path,
    coord_permutation: str = "2,0,1",
    box_permutation: Optional[str] = None,  # None means same as coord
    sh_transform_mode: str = "standard",
    sh_l1_override: Optional[str] = None,
    mirror_x: bool = False,
    cutoff_radius: Optional[float] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Snapshot:
    """
    Load OpenMX snapshot with configurable transformations.

    This function allows systematic exploration of transformation variants
    to identify potential bugs in the data loading pipeline.

    Args:
        matrix_path: Path to .scfout file
        info_path: Path to .info.out file
        coord_permutation: Coordinate permutation ("2,0,1", "1,2,0", "none")
        box_permutation: Box permutation (None = same as coords, or "none")
        sh_transform_mode: SH basis transform ("standard", "inverse", "none")
        sh_l1_override: Override for l=1 orbitals ("2,0,1", "1,2,0", "none", None)
        mirror_x: Mirror x coordinate
        cutoff_radius: Optional distance cutoff
        device: torch device
        dtype: torch dtype

    Returns:
        Snapshot with applied transformations
    """
    # Parse info file
    info = parse_info_out(info_path, dtype)
    atoms = info.elements
    if not atoms:
        raise RuntimeError("Info-file does not contain <coordinates.forces>")

    orb_cfg = OrbitalIrrepConfig.from_dict(info.orbital_set)

    # Parse matrix file (stays in OpenMX basis)
    snap = parse_openmx_scfout(
        matrix_path,
        atoms,
        orb_cfg,
        convention="openmx",  # Don't convert yet
        symmetrize_density=True,  # Keep standard symmetrization
    )

    # Apply custom SH basis transformation
    l_overrides = {}
    if sh_l1_override is not None:
        l_overrides[1] = sh_l1_override

    converter = ConfigurableBasisConverter(
        orb_cfg,
        mode=sh_transform_mode,
        l_overrides=l_overrides,
        device=device,
    )

    ham = converter.matrix_transform(snap.hamiltonian)
    ovl = converter.matrix_transform(snap.overlap)
    den = converter.matrix_transform(snap.density)

    # Apply coordinate transformations
    positions = apply_coordinate_transform(info.positions, coord_permutation, mirror_x)

    # Box permutation (default to same as coordinates)
    if box_permutation is None:
        box_perm = coord_permutation
    else:
        box_perm = box_permutation

    box = (
        apply_coordinate_transform(
            info.box, box_perm, mirror_x=False  # Don't mirror box
        )
        if info.box.numel()
        else None
    )

    # Create snapshot
    snap_transformed = Snapshot(
        hamiltonian=ham,
        overlap=ovl,
        density=den,
        positions=positions,
        box=box,
        matrix_path=matrix_path,
        info_path=info_path,
    )

    # Apply cutoff and canonicalize
    if cutoff_radius is not None:
        snap_transformed = snap_transformed.filter_by_distance(cutoff_radius)

    snap_transformed = snap_transformed.canonicalize_edges()

    return snap_transformed


def get_transform_config_str(
    coord_permutation: str,
    box_permutation: Optional[str],
    sh_transform_mode: str,
    sh_l1_override: Optional[str],
    mirror_x: bool,
) -> str:
    """
    Get a compact string describing the transformation configuration.
    Useful for run names and logging.
    """
    parts = []
    parts.append(f"coord{coord_permutation.replace(',', '')}")

    if box_permutation is None:
        parts.append("box_same")
    else:
        parts.append(f"box{box_permutation.replace(',', '')}")

    parts.append(f"sh{sh_transform_mode}")

    if sh_l1_override is not None:
        parts.append(f"l1_{sh_l1_override.replace(',', '')}")

    if mirror_x:
        parts.append("mirrorX")

    return "_".join(parts)
