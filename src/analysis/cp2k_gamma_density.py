"""Transfer finite-support Gamma AO densities without rounding occupations.

Source and target matrices are restricted total densities. Unrecorded source
atom pairs are explicitly zero, as in the supervised block representation.
This is not a claim to recover the full nonlocal KS density operator.
"""

import copy
import numpy as np
from scipy.linalg import eigh
from data.cp2k_restart import CP2KImageDensity


def finite_gamma_density(dump, *, tolerance=1e-10):
    sections = [s for s in dump.sections if s.label.startswith("Density matrix")]
    labels = [s.label for s in sections]
    if labels not in (
        ["Density matrix spin=0"],
        ["Density matrix spin=0", "Density matrix spin=0", "Density matrix spin=1"],
    ):
        raise ValueError(
            f"Unsupported restricted real/imaginary section order: {labels}"
        )
    if any(np.max(abs(b)) > 1e-12 for s in sections[1:] for b in s.blocks.values()):
        raise ValueError("A real Gamma density is required")
    offsets = np.cumsum([0, *dump.block_sizes])
    n = offsets[-1]
    p = np.zeros((n, n))
    seen = {}
    maximum = 0.0
    for (i, j, _), block in sections[0].blocks.items():
        key = (i, j)
        if key in seen:
            maximum = max(maximum, float(np.max(abs(block - seen[key]))))
        else:
            seen[key] = block
            p[offsets[i - 1] : offsets[i], offsets[j - 1] : offsets[j]] = 2 * block
    if maximum > tolerance:
        raise ValueError("Density images are not repeated Gamma blocks")
    error = float(np.max(abs(p - p.T)))
    if error > 1e-7:
        raise ValueError(f"Native density Hermiticity error {error}")
    return (p + p.T) / 2, dict(
        spin_factor=2,
        recorded_atom_pairs=len(seen),
        missing_atom_pairs=dump.atom_count**2 - len(seen),
        image_repetition_error=maximum,
        hermiticity_error=error,
        missing_pair_policy="zero, finite-support supervised representation; no H completion",
    )


def map_density(p, overlap, *, cross=None, permutation=None, electrons, threshold=1e-5):
    if (cross is None) == (permutation is None):
        raise ValueError("Choose exactly one transfer map")
    s = (overlap + overlap.T) / 2
    v, u = eigh(s)
    keep = v > threshold
    if cross is not None:
        w = u[:, keep] / np.sqrt(v[keep])
        transform = w @ (w.T @ cross)
        raw = transform @ p @ transform.T
    else:
        raw = p[np.ix_(permutation, permutation)]
    raw = (raw + raw.T) / 2
    count = float(np.sum(raw * s))
    if count <= 0 or not np.isfinite(count):
        raise ValueError("Mapped density has invalid target charge")
    factor = electrons / count
    result = raw * factor
    # Diagnostics only: no occupation clipping or occupied-space projector.
    root = (u * np.sqrt(np.maximum(v, 0))) @ u.T
    occupations = eigh(root @ result @ root, eigvals_only=True)
    return result, dict(
        raw_target_electrons=count,
        charge_factor=factor,
        final_target_electrons=float(np.sum(result * s)),
        metric_eigenvalue_min=float(occupations.min()),
        metric_eigenvalue_max=float(occupations.max()),
        discarded_target_directions=int((~keep).sum()),
        occupation_policy="unrounded density; scalar charge normalization only; no PSD repair",
    )


def gamma_template_layout(restart, *, block_size=13, tolerance=1e-9):
    """Certify repeated Gamma blocks and retain whole-atom storage masks.

    Only template support/metadata enter a new guess. Whole-block masks commute
    with independent AO rotations. Template density values are used solely for
    roundtrip/interface certification, never mixed into a transferred density.
    """
    if len(restart.spins) != 1 or restart.nao != restart.natom * block_size:
        raise ValueError("Restricted complete s2p2d1 atom blocks required")
    n = restart.natom
    b = block_size
    common = np.zeros((n, n, b, b))
    seen = np.zeros((n, n), bool)
    masks = []
    error = 0.0
    minimum = np.inf
    for image in restart.spins[0]:
        blocks = image.matrix.reshape(n, b, n, b).transpose(0, 2, 1, 3)
        norms = np.linalg.norm(blocks, axis=(2, 3))
        mask = norms > 1e-18
        if mask.any():
            minimum = min(minimum, float(norms[mask].min()))
        old = mask & seen
        if old.any():
            error = max(error, float(np.max(abs(blocks[old] - common[old]))))
        new = mask & ~seen
        common[new] = blocks[new]
        seen |= mask
        masks.append(mask)
    if error > tolerance or minimum < 1e-12:
        raise ValueError(
            f"Uncertified Gamma storage: repetition={error}, minimum block={minimum}"
        )
    stored = int(seen.sum())
    # Canonical native storage can imply the reverse atom block instead of
    # writing it explicitly. At real Gamma its value is the transpose.
    for i, j in zip(*np.nonzero(~seen & seen.T)):
        common[i, j] = common[j, i].T
        seen[i, j] = True
    p = common.transpose(0, 2, 1, 3).reshape(restart.nao, restart.nao)
    if np.max(abs(p - p.T)) > tolerance:
        raise ValueError("Native Gamma union is not Hermitian")
    return (
        p,
        np.asarray(masks),
        dict(
            max_repetition_error=error,
            minimum_stored_block_norm=minimum,
            images=len(masks),
            stored_atom_pairs=stored,
            implied_reverse_pairs=int(seen.sum()) - stored,
            mask_policy="whole atom blocks from native template; implied Hermitian reverses restored",
        ),
    )


def pack_gamma(template, p, masks, *, block_size=13):
    n = template.natom
    b = block_size
    if p.shape != (template.nao, template.nao) or masks.shape != (
        len(template.spins[0]),
        n,
        n,
    ):
        raise ValueError("Gamma payload/layout dimensions differ")
    blocks = p.reshape(n, b, n, b).transpose(0, 2, 1, 3)
    result = copy.copy(template)
    result.spins = [[]]
    for image, mask in zip(template.spins[0], masks):
        value = (blocks * mask[:, :, None, None]).transpose(0, 2, 1, 3).reshape(p.shape)
        result.spins[0].append(CP2KImageDensity(image.cell, value))
    return result
