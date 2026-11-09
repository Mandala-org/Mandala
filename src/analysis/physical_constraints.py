"""Unrepaired, metric-aware operator diagnostics for a finite periodic support.

The density is the contravariant AO kernel, not its ordinary eigenvalue
spectrum. No eigenvalue clipping, overlap jitter or charge rescaling is used.
All dense algebra is complex128; predictions can still originate in float32.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import torch


def hermitian(a):
    return (a + a.mH) * 0.5


def _norm(a):
    return float(torch.linalg.vector_norm(a))


def hermiticity(a):
    error = a - a.mH
    return {
        "relative_frobenius": _norm(error) / max(_norm(a), 1e-30),
        "max_absolute": float(error.abs().max()),
    }


def uniform_kpoints(mesh):
    """Gamma-centered fractional mesh, with normalized equal weights."""
    if len(mesh) != 3 or min(mesh) < 1:
        raise ValueError("Expected three positive mesh sizes")
    return [
        list(k)
        for k in itertools.product(
            *[[j / n if j <= n // 2 else j / n - 1 for j in range(n)] for n in mesh]
        )
    ]


class BlockFourier:
    """Assemble one k point at a time, without an Nshift*nao**2 buffer.

    The sign agrees with core.periodic_fourier: exp(+2 pi i k.L).
    Missing blocks mean zero; reverse edges are never fabricated here.
    """

    def __init__(self, matrix, *, device="cpu"):
        self.device = torch.device(device)
        dims = [matrix.orbital_cfg.block_dims(f"{el}-{el}")[0] for el in matrix.atoms]
        offsets = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)
        self.dim = offsets[-1]
        self.parts = []
        seen = set()
        for pair, edges in matrix.pair_edges.items():
            edge = edges.detach().cpu().T.long()
            keys = [tuple(x) for x in edge.tolist()]
            if len(keys) != len(set(keys)) or seen.intersection(keys):
                raise ValueError("Duplicate periodic block keys")
            seen.update(keys)
            values = matrix.pair_blocks[pair].detach().to(self.device, torch.complex128)
            ni, nj = values.shape[-2:]
            if any(dims[i] != ni or dims[j] != nj for i, j in edge[:, 3:].tolist()):
                raise ValueError("Block shape does not match atom orbital dimensions")
            row = torch.tensor([offsets[i] for i in edge[:, 3]], device=self.device)
            col = torch.tensor([offsets[j] for j in edge[:, 4]], device=self.device)
            indices = (
                (
                    row[:, None, None]
                    + torch.arange(ni, device=self.device)[None, :, None]
                )
                * self.dim
                + col[:, None, None]
                + torch.arange(nj, device=self.device)[None, None, :]
            )
            self.parts.append(
                (edge[:, :3].to(self.device, torch.float64), indices.flatten(), values)
            )
        self.reverse_missing = sum(
            (-x, -y, -z, j, i) not in seen for x, y, z, i, j in seen
        )

    def at(self, fractional_k):
        k = torch.as_tensor(fractional_k, device=self.device, dtype=torch.float64)
        out = torch.zeros(
            self.dim * self.dim, device=self.device, dtype=torch.complex128
        )
        for shifts, indices, values in self.parts:
            phase = torch.exp(2j * math.pi * (shifts @ k))
            out.index_add_(0, indices, (values * phase[:, None, None]).flatten())
        return out.reshape(self.dim, self.dim)


@torch.no_grad()
def diagnose_kpoint(
    h,
    s,
    d=None,
    *,
    max_occupation=2.0,
    occupation_tolerance=1e-5,
    overlap_atol=1e-8,
    overlap_rtol=1e-8,
):
    """Raw Hermiticity followed by explicitly labelled Hermitian diagnostics.

    F=S^-1/2 H S^-1/2, P=S^1/2 D S^1/2. The commutator is equivalent
    to H D S-S D H=0 and its norm is independent of a constant energy gauge.
    Idempotency is only a zero-temperature diagnostic, not a finite-T demand.
    Private arrays support a subsequently fitted *global* Fermi level.
    """
    if max_occupation <= 0 or min(occupation_tolerance, overlap_atol, overlap_rtol) < 0:
        raise ValueError("Invalid diagnostic tolerances/occupation")
    matrices = {"hamiltonian": h, "overlap": s}
    if d is not None:
        matrices["density"] = d
    matrices = {key: a.to(torch.complex128) for key, a in matrices.items()}
    if any(
        a.shape != s.shape or a.ndim != 2 or a.shape[0] != a.shape[1]
        for a in matrices.values()
    ):
        raise ValueError("All operators must be equally sized square matrices")
    if not all(bool(torch.isfinite(a).all()) for a in matrices.values()):
        return {"status": "nonfinite_operator"}
    result = {
        "status": "ok",
        "raw_hermiticity": {key: hermiticity(a) for key, a in matrices.items()},
    }
    h, s = hermitian(matrices["hamiltonian"]), hermitian(matrices["overlap"])
    se, su = torch.linalg.eigh(s)
    minimum, maximum = float(se[0]), float(se[-1])
    threshold = max(overlap_atol, overlap_rtol * max(abs(maximum), abs(minimum)))
    _, info = torch.linalg.cholesky_ex(s)
    result["overlap"] = {
        "lambda_min": minimum,
        "lambda_max": maximum,
        "condition_number": maximum / minimum if minimum > 0 else None,
        "nonpositive_eigenvalues": int((se <= 0).sum()),
        "below_threshold_eigenvalues": int((se <= threshold).sum()),
        "positive_definite": minimum > 0,
        "well_conditioned_positive_definite": minimum > threshold,
        "threshold": threshold,
        "cholesky_failed_without_jitter": bool(info),
    }
    if minimum <= threshold or bool(info):
        result["status"] = "invalid_overlap_metric"
        return result
    sinv = (su * se.rsqrt()[None, :]) @ su.mH
    f = hermitian(sinv @ h @ sinv)
    energy, orbitals = torch.linalg.eigh(f)
    result["_energy_hartree"] = energy.cpu()
    result["eigensolve"] = {
        "finite": bool(torch.isfinite(energy).all()),
        "energy_min_hartree": float(energy[0]),
        "energy_max_hartree": float(energy[-1]),
    }
    if d is None:
        return result
    sqrt = (su * se.sqrt()[None, :]) @ su.mH
    p = hermitian(sqrt @ hermitian(matrices["density"]) @ sqrt)
    occupations = torch.linalg.eigvalsh(p)
    negative, excess = (
        (-occupations).clamp_min(0),
        (occupations - max_occupation).clamp_min(0),
    )
    result["density"] = {
        "occupation_min": float(occupations[0]),
        "occupation_max": float(occupations[-1]),
        "negative_count": int((occupations < -occupation_tolerance).sum()),
        "overoccupied_count": int(
            (occupations > max_occupation + occupation_tolerance).sum()
        ),
        "orbital_count": occupations.numel(),
        "negative_mass": float(negative.sum()),
        "overoccupation_mass": float(excess.sum()),
        "electron_trace": float(occupations.sum()),
        "zero_temperature_idempotency_relative": _norm(p @ p - max_occupation * p)
        / max(max_occupation * _norm(p), 1e-30),
    }
    f_centered = f - torch.eye(f.shape[0], device=f.device) * (
        torch.trace(f).real / f.shape[0]
    )
    commutator = f_centered @ p - p @ f_centered
    pe = orbitals.mH @ p @ orbitals
    diagonal = pe.diagonal().real
    result["consistency"] = {
        "commutator_relative": _norm(commutator)
        / max(_norm(f_centered) * _norm(p), 1e-30),
        "commutator_frobenius_hartree": _norm(commutator),
        "density_offdiagonal_in_h_eigenbasis_relative": _norm(
            pe - torch.diag_embed(diagonal)
        )
        / max(_norm(p), 1e-30),
        "band_energy_trace_hartree": float(torch.trace(f @ p).real),
    }
    result["_density_diagonal_in_h_basis"] = diagonal.cpu()
    result["_density_norm_squared"] = _norm(p) ** 2
    return result


def fermi_consistency(
    records, *, num_electrons, temperature_kelvin, max_occupation=2.0
):
    """Compare D to f(H,S) at fixed physical charge and one common chemical potential.

    Uniform k weights are used. An inverted population can commute with H;
    this extra test detects it. Finite temperature is mandatory here.
    """
    if not records or temperature_kelvin <= 0:
        raise ValueError("Need records and a positive temperature")
    if any("_energy_hartree" not in r for r in records):
        return {"status": "unavailable_invalid_overlap"}
    energies = torch.stack([r["_energy_hartree"] for r in records])
    if not 0 < num_electrons < max_occupation * energies.shape[1]:
        raise ValueError("Electron count must lie inside orbital capacity")
    kt = 3.166811563e-6 * temperature_kelvin
    low, high = float(energies.min()) - 100 * kt, float(energies.max()) + 100 * kt
    # Balance holes and particles around a zero-temperature filling. Directly
    # subtracting N from sum(f) loses the exponentially small charge imbalance
    # in an insulating gap and pins mu arbitrarily to a band edge.
    sorted_energy = energies.flatten().sort().values
    target_total = num_electrons * len(records)
    filled = int(target_total / max_occupation)
    for _ in range(120):
        mu = (low + high) / 2
        holes = max_occupation * torch.sigmoid((sorted_energy[:filled] - mu) / kt)
        particles = max_occupation * torch.sigmoid((mu - sorted_energy[filled:]) / kt)
        imbalance = (filled * max_occupation - target_total) + float(
            particles.sum() - holes.sum()
        )
        if imbalance == 0:
            break
        if imbalance < 0:
            low = mu
        else:
            high = mu
    occ = max_occupation * torch.sigmoid((mu - energies) / kt)
    result = {
        "status": "ok",
        "chemical_potential_hartree": mu,
        "temperature_kelvin": temperature_kelvin,
        "reconstructed_electron_count": float(occ.sum(dim=1).mean()),
    }
    if all("_density_diagonal_in_h_basis" in r for r in records):
        diagonal = torch.stack([r["_density_diagonal_in_h_basis"] for r in records])
        norm_squared = sum(r["_density_norm_squared"] for r in records)
        target_norm_squared = float(occ.square().sum())
        delta_squared = max(
            0.0, norm_squared + target_norm_squared - 2 * float((occ * diagonal).sum())
        )
        result["density_relative_frobenius"] = math.sqrt(
            delta_squared / max(target_norm_squared, 1e-30)
        )
        result["diagonal_occupation_mae"] = float((diagonal - occ).abs().mean())
    return result


def summarize_records(records):
    """Keep denominators and failure counts explicit; never omit bad k points."""
    values = defaultdict(list)

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if not key.startswith("_"):
                    visit(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                values[prefix].append(float(value))
        elif isinstance(value, bool):
            values[prefix].append(float(value))

    for record in records:
        visit("", record)
    return {
        "kpoint_count": len(records),
        "invalid_kpoint_count": sum(r["status"] != "ok" for r in records),
        "metrics": {
            key: {
                "count": len(v),
                "mean": sum(v) / len(v),
                "min": min(v),
                "max": max(v),
                "p95": float(
                    torch.quantile(torch.tensor(v, dtype=torch.float64), 0.95)
                ),
            }
            for key, v in sorted(values.items())
        },
    }


def public_record(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}
