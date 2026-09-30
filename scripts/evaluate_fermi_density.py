"""Compare direct Si density with finite-temperature Hamiltonian reconstruction.

Native OpenMX spin=0 normalization is requested explicitly, independently of
whether the checkout's default loader symmetrization includes a spin factor.
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq
from scipy.special import expit, log_expit, logsumexp
from scipy.constants import physical_constants
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot
from net.common import Config
from utils.units import HARTREE_TO_EV

KB_HA = physical_constants["Boltzmann constant in eV/K"][0] / HARTREE_TO_EV


def occupations(energies, temperature, nocc):
    """One chemical potential for the entire equally weighted k mesh."""
    if temperature == 0:
        gap = energies[:, nocc].min() - energies[:, nocc - 1].max()
        if gap <= 0:
            raise ValueError(
                "Integer-occupation baseline requires a positive global gap"
            )
        mu = (energies[:, nocc].min() + energies[:, nocc - 1].max()) / 2
        return (energies < mu).astype(float), float(mu)
    kt = KB_HA * temperature

    # Balance conduction electrons against valence holes in log space.
    # Subtracting the total occupation from nocc loses both at low T.
    def charge_balance(m):
        return logsumexp(log_expit((m - energies[:, nocc:]) / kt)) - logsumexp(
            log_expit((energies[:, :nocc] - m) / kt)
        )

    mu = brentq(
        charge_balance, energies.min() - 50 * kt, energies.max() + 50 * kt, xtol=1e-14
    )
    f = expit((mu - energies) / kt)
    assert abs(f.sum(axis=1).mean() - nocc) < 1e-9
    return f, float(mu)


def fourier(m, k):
    dims = [m.orbital_cfg.block_dims(f"{a}-{a}")[0] for a in m.atoms]
    off = np.r_[0, np.cumsum(dims)]
    out = np.zeros((len(k), off[-1], off[-1]), dtype=complex)
    for edge, (key, index) in m.lookup.items():
        *shift, i, j = edge
        out[:, off[i] : off[i + 1], off[j] : off[j + 1]] += (
            np.exp(2j * np.pi * (k @ shift))[:, None, None]
            * m.pair_blocks[key][index].numpy()
        )
    return (out + out.conj().transpose(0, 2, 1)) / 2


def inverse(dk, k, support):
    dims = [support.orbital_cfg.block_dims(f"{a}-{a}")[0] for a in support.atoms]
    off = np.r_[0, np.cumsum(dims)]
    blocks = {key: torch.zeros_like(v) for key, v in support.pair_blocks.items()}
    for edge, (key, index) in support.lookup.items():
        *shift, i, j = edge
        block = np.einsum(
            "k,kij->ij",
            np.exp(-2j * np.pi * (k @ shift)),
            dk[:, off[i] : off[i + 1], off[j] : off[j + 1]],
        ) / len(k)
        assert abs(block.imag).max() < 1e-10
        blocks[key][index] = torch.from_numpy(block.real.copy())
    return support._replace_pair_blocks(blocks, basis=support.basis)


def trace(a, b):
    value = 0.0
    for (x, y, z, i, j), (key, index) in a.lookup.items():
        match = b.lookup.get((-x, -y, -z, j, i))
        if match:
            value += (
                (a.pair_blocks[key][index] * b.pair_blocks[match[0]][match[1]].T)
                .sum()
                .item()
            )
    return value


def errors(a, b):
    differences = []
    for edge in a.lookup.keys() | b.lookup.keys():

        def block(m):
            match = m.lookup.get(edge)
            return m.pair_blocks[match[0]][match[1]].numpy() if match else 0.0

        differences.append(np.ravel(block(a) - block(b)))
    diff = np.concatenate(differences)
    return dict(mae=float(abs(diff).mean()), rmse=float(np.sqrt(np.mean(diff**2))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--temperature-k",
        type=float,
        default=None,
        help="Default: electronic temperature parsed from reference Si.out; never inferred from sigma alone",
    )
    p.add_argument(
        "--temperatures-k",
        type=float,
        nargs="+",
        default=None,
        help="Temperature sweep; reuse the same eigenpairs",
    )
    p.add_argument("--mesh", type=int, default=8)
    a = p.parse_args()
    if a.temperature_k is not None and a.temperatures_k is not None:
        p.error("Choose --temperature-k or --temperatures-k")
    requested = a.temperatures_k or (
        [] if a.temperature_k is None else [a.temperature_k]
    )
    if a.mesh < 1 or any(not np.isfinite(t) or t <= 0 for t in requested):
        p.error("mesh and finite temperature must be positive")
    torch.set_num_threads(4)
    refdir = ROOT / "data/paper/paper_Si_perturbed_val_090"
    preddir = (
        ROOT
        / "model_reports/paper_Si_perturbed_best_val090/run_assets/paper_Si_perturbed_best_val090"
    )
    source_temperature = float(
        re.search(
            r"^scf\.ElectronicTemperature\s+(\S+)",
            (refdir / "Si.out").read_text(),
            re.M,
        ).group(1)
    )
    temperatures = requested or [source_temperature]
    snap = Snapshot.from_openmx(
        refdir / "HS.out",
        refdir / "Si.out",
        convention="e3nn",
        symmetrize_density=False,
        cfg=Config(verbosity=0),
    )

    def physical(m):
        m = m._replace_pair_blocks(
            {key: v.detach().cpu().double() for key, v in m.pair_blocks.items()},
            basis=m.basis,
        )
        return 0.5 * (m + m.transpose())

    dr, sr, hr = map(physical, (snap.density, snap.overlap, snap.hamiltonian))
    dp, sp, hp = [
        physical(BlockMatrix.load(preddir / f"pred_{name}.pt"))
        for name in ("density", "overlap", "hamiltonian")
    ]
    dp = 0.5 * dp  # Explicit one-time legacy doubled-target conversion.
    for m in (dp, sp, hp):
        assert (
            m.atoms == dr.atoms
            and m.basis == dr.basis
            and m.orbital_cfg.to_dict() == dr.orbital_cfg.to_dict()
        )
    assert all(x == "Si" for x in dr.atoms)
    nocc = 2 * len(dr.atoms)
    k = np.stack(
        np.meshgrid(*([np.arange(a.mesh) / a.mesh] * 3), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    eref = 2 * trace(dr, hr)

    def row(label, m):
        return dict(
            method=label,
            **errors(m, dr),
            band_energy_ref_h_abs_error_ev=abs(2 * trace(m, hr) - eref) * HARTREE_TO_EV,
            electrons_reference_s=2 * trace(m, sr),
        )

    rows = [row("Direct density prediction", dp)]
    for label, hm, sm in [
        ("Reference H + reference S", hr, sr),
        ("Predicted H + reference S", hp, sr),
        ("Predicted H + predicted S", hp, sp),
    ]:
        print(f"Diagonalizing {label}: {len(k)} k points", flush=True)
        hk, sk = fourier(hm, k), fourier(sm, k)
        evals = np.empty((len(k), hk.shape[-1]))
        coeff = np.empty_like(hk)
        max_residual = 0.0
        for idx in range(len(k)):
            e, c = la.eigh(hk[idx], sk[idx])
            evals[idx] = e
            coeff[idx] = c
            residual = la.norm(hk[idx] @ c - (sk[idx] @ c) * e) / la.norm(hk[idx] @ c)
            assert residual < 1e-10
            assert abs(c.conj().T @ sk[idx] @ c - np.eye(len(e))).max() < 1e-10
            max_residual = max(max_residual, residual)
        baseline = None
        for temp in [0.0, *temperatures]:
            f, mu = occupations(evals, temp, nocc)
            dk = (coeff * f[:, None, :]) @ coeff.conj().transpose(0, 2, 1)
            m = inverse(dk, k, dr)
            if baseline is None:
                baseline = m
            r = row(label, m)
            r.update(
                temperature_k=temp,
                chemical_potential_ha=mu,
                global_gap_ev=float(
                    (evals[:, nocc].min() - evals[:, nocc - 1].max()) * HARTREE_TO_EV
                ),
                thermal_electrons_per_cell=float(2 * f[:, nocc:].sum(axis=1).mean()),
                occupation_electrons=float(2 * f.sum(axis=1).mean()),
                max_eigenproblem_residual=float(max_residual),
                change_from_zero_temperature=errors(m, baseline),
            )
            rows.append(r)
    inputs = [
        refdir / "HS.out",
        refdir / "Si.out",
        refdir / "Si.cif",
        *(
            preddir / f"pred_{name}.pt"
            for name in ("density", "overlap", "hamiltonian")
        ),
    ]
    result = dict(
        snapshot="Si validation 090 / scale_1",
        temperatures_k=temperatures,
        temperature_source=(
            "Explicit CLI override" if requested else "Si.out scf.ElectronicTemperature"
        ),
        reference_electronic_temperature_k=source_temperature,
        perturbation_sigma_angstrom=0.02,
        scale_to_temperature_calibration=None,
        mesh=a.mesh,
        mesh_convention="Uniform Gamma-centered arange(N)/N; equal weights",
        density_convention="Native per-spin density; explicit 0.5 conversion of legacy prediction; no default loader spin factor",
        rows=rows,
        input_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
            if path.exists()
        },
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Finite-temperature Hamiltonian density: Si validation 090",
        "",
        "Temperatures: "
        + ", ".join(f"{t:g}" for t in temperatures)
        + " K. Source: "
        + result["temperature_source"]
        + ".",
        "",
        "The scale_1 generator specifies independent Gaussian Cartesian displacement sigma=0.02 Å (nominal vector RMS sqrt(3)*sigma=0.03464 Å). No temperature calibration is recorded. Displacement amplitude alone does not specify either ionic or electronic temperature. Temperatures are explicitly supplied for a sweep or taken from the recorded electronic temperature by default.",
        "",
        f"Solve HC=SC E with C†SC=I on a uniform Gamma-centered {a.mesh}³ mesh. Use f=1/(1+exp((E-mu)/(kB*T))) and D(k)=C f C†. A single chemical potential per H/S model is fitted to 32 electrons across the entire mesh, not separately at each k point. At low temperature, solve electron/hole balance in log space to avoid loss of precision from subtracting nearly equal total charges. The zero-temperature baseline fills 16 states per spin and requires a positive global gap. Inverse Fourier transform onto the original periodic support.",
        "",
        "Reference density is explicitly loaded without the checkout-dependent default spin factor and Hermitian-averaged. Legacy direct predictions are scaled by 0.5 once. MAE and RMSE use scalar entries over the union of supports. Band-energy errors hold reference H fixed and use 2 Tr(D H_ref), eV/cell.",
        "",
        "| Method | T (K) | MAE | RMSE | Band-energy error (eV/cell) | Thermal conduction electrons/cell |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = f"{r['temperature_k']:g}" if "temperature_k" in r else "—"
        ne = (
            f"{r['thermal_electrons_per_cell']:.6g}"
            if "thermal_electrons_per_cell" in r
            else "—"
        )
        lines.append(
            f"| {r['method']} | {t} | {r['mae']:.9g} | {r['rmse']:.9g} | {r['band_energy_ref_h_abs_error_ev']:.9g} | {ne} |"
        )
    lines += [
        "",
        "All eigenproblem and S-orthogonality residuals are checked below 1e-10; mesh-integrated occupations agree with 32 electrons to 2e-9. Inverse Fourier imaginary parts are checked below 1e-10. This comparison changes occupations at fixed Hamiltonian; it is not a new self-consistent DFT calculation.",
        "",
        "Reproduce:",
        "",
        "```bash",
        f'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mandala-venv/bin/python scripts/evaluate_fermi_density.py --temperatures-k {" ".join(f"{t:g}" for t in temperatures)} --mesh {a.mesh} --output-dir studies/mcweeny/fermi_density',
        "```",
        "",
    ]
    (a.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
