"""Gamma occupations and occupied-Hamiltonian density comparison for saved Si models."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import scipy.linalg as la
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot
from net.common import Config
from analysis.density_purification import density_errors
from core.sparse_math import (
    build_trace_alignment,
    trace_matmul_sparse_block_matrix_aligned,
)
from utils.units import HARTREE_TO_EV


def gamma(matrix):
    raw = matrix.to_dense().numpy()
    return (raw + raw.T) / 2


def metrics(d, ref):
    diff = d - ref
    return dict(
        mae=float(np.abs(diff).mean()),
        rmse=float(np.sqrt(np.mean(abs(diff) ** 2))),
        relative_frobenius=float(la.norm(diff) / la.norm(ref)),
    )


def occupied_density(h, s, nocc):
    e, c = la.eigh(h, s)
    d = c[:, :nocc] @ c[:, :nocc].conj().T
    residual = la.norm(h @ c - (s @ c) * e) / max(la.norm(h @ c), 1e-30)
    orth = float(np.max(np.abs(c.conj().T @ s @ c - np.eye(len(e)))))
    assert residual < 1e-10 and orth < 1e-10
    assert abs(np.trace(d @ s) - nocc) < 1e-9
    return (
        d,
        e,
        dict(eigenproblem_relative_residual=float(residual), s_orthogonality_max=orth),
    )


def fourier(matrix, kpoints):
    dims = [matrix.orbital_cfg.block_dims(f"{a}-{a}")[0] for a in matrix.atoms]
    offsets = np.r_[0, np.cumsum(dims)]
    out = np.zeros((len(kpoints), offsets[-1], offsets[-1]), dtype=np.complex128)
    for edge, (key, index) in matrix.lookup.items():
        *shift, i, j = edge
        phase = np.exp(2j * np.pi * (kpoints @ shift))
        out[:, offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] += (
            phase[:, None, None] * matrix.pair_blocks[key][index].numpy()
        )
    return (out + out.conj().transpose(0, 2, 1)) / 2


def inverse_on_support(dk, kpoints, support):
    dims = [support.orbital_cfg.block_dims(f"{a}-{a}")[0] for a in support.atoms]
    offsets = np.r_[0, np.cumsum(dims)]
    blocks = {
        key: torch.zeros_like(value) for key, value in support.pair_blocks.items()
    }
    imaginary_max = 0.0
    for edge, (key, index) in support.lookup.items():
        *shift, i, j = edge
        phase = np.exp(-2j * np.pi * (kpoints @ shift))
        block = np.einsum(
            "k,kij->ij",
            phase,
            dk[:, offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]],
        ) / len(kpoints)
        imaginary_max = max(imaginary_max, float(abs(block.imag).max()))
        blocks[key][index] = torch.from_numpy(block.real.copy())
    assert imaginary_max < 1e-10
    return support._replace_pair_blocks(blocks, basis=support.basis), imaginary_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--mesh", type=int, default=8)
    args = parser.parse_args()
    if args.mesh < 1:
        parser.error("--mesh must be positive")
    torch.set_num_threads(4)
    case = json.loads((ROOT / "studies/mcweeny/silicon_cases.json").read_text())[
        "cases"
    ][args.case_index]
    reference = Snapshot.from_openmx(
        ROOT / case["matrix_path"],
        ROOT / case["info_path"],
        convention="e3nn",
        cfg=Config(verbosity=0),
    )

    def double(m):
        return m._replace_pair_blocks(
            {k: v.detach().cpu().double() for k, v in m.pair_blocks.items()},
            basis=m.basis,
        )

    dr, sr, hr = map(
        double, (reference.density, reference.overlap, reference.hamiltonian)
    )
    pred_dir = ROOT / case["prediction_dir"]
    dp, sp, hp = [
        double(BlockMatrix.load(pred_dir / f"pred_{name}.pt"))
        for name in ("density", "overlap", "hamiltonian")
    ]
    dp = case["saved_density_scale"] * dp
    for m in (dp, sp, hp):
        assert (
            m.atoms == dr.atoms
            and m.basis == dr.basis
            and m.orbital_cfg.to_dict() == dr.orbital_cfg.to_dict()
        )
    assert all(a == "Si" for a in dr.atoms)
    nocc = 2 * len(dr.atoms)  # Four valence electrons per Si, two spin channels.
    d, s, h = map(gamma, (dr, sr, hr))
    se, su = la.eigh(s)
    assert se.min() > 0
    sqrt_s = (su * np.sqrt(se)) @ su.T
    invsqrt_s = (su * (1 / np.sqrt(se))) @ su.T
    occ, u = la.eigh(sqrt_s @ d @ sqrt_s)
    c = invsqrt_s @ u
    ds_residual = float(la.norm(d @ s @ c - c * occ) / la.norm(c))
    assert ds_residual < 1e-10
    assert abs(occ.sum() - np.trace(d @ s)) < 1e-10
    variants = [
        ("Reference H + reference S", hr, sr),
        ("Predicted H + reference S", hp, sr),
        ("Predicted H + predicted S", hp, sp),
    ]
    gamma_rows = [dict(method="Direct density prediction", **metrics(gamma(dp), d))]
    for label, hm, sm in variants:
        recon, energies, checks = occupied_density(gamma(hm), gamma(sm), nocc)
        gamma_rows.append(
            dict(
                method=label,
                **metrics(recon.real, d),
                **checks,
                gap_ev=float((energies[nocc] - energies[nocc - 1]) * HARTREE_TO_EV),
            )
        )
    mesh_axis = np.arange(args.mesh) / args.mesh
    kpoints = np.stack(
        np.meshgrid(mesh_axis, mesh_axis, mesh_axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    def sparse_row(label, prediction):
        alignment = build_trace_alignment(prediction, hr)
        energy = (
            2
            * trace_matmul_sparse_block_matrix_aligned(prediction, hr, alignment).item()
        )
        energy_ref = (
            2
            * trace_matmul_sparse_block_matrix_aligned(
                dr, hr, build_trace_alignment(dr, hr)
            ).item()
        )
        electrons = (
            2
            * trace_matmul_sparse_block_matrix_aligned(
                prediction, sr, build_trace_alignment(prediction, sr)
            ).item()
        )
        return dict(
            method=label,
            **density_errors(prediction, dr),
            band_energy_ref_h_abs_error_ev=abs(energy - energy_ref) * HARTREE_TO_EV,
            electrons=electrons,
        )

    sparse_rows = [sparse_row("Direct density prediction", 0.5 * (dp + dp.transpose()))]
    for label, hm, sm in variants:
        print(f"Reconstructing {label}: {len(kpoints)} k-points", flush=True)
        hk, sk = fourier(hm, kpoints), fourier(sm, kpoints)
        dk = np.empty_like(hk)
        vbm, cbm, max_residual, max_orth = -np.inf, np.inf, 0.0, 0.0
        for index in range(len(kpoints)):
            dk[index], energies, checks = occupied_density(hk[index], sk[index], nocc)
            vbm, cbm = max(vbm, energies[nocc - 1]), min(cbm, energies[nocc])
            max_residual = max(max_residual, checks["eigenproblem_relative_residual"])
            max_orth = max(max_orth, checks["s_orthogonality_max"])
        recon, imaginary = inverse_on_support(dk, kpoints, dr)
        sparse_rows.append(
            dict(
                **sparse_row(label, recon),
                global_gap_ev=float((cbm - vbm) * HARTREE_TO_EV),
                eigenproblem_relative_residual=max_residual,
                s_orthogonality_max=max_orth,
                imaginary_max=imaginary,
            )
        )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out / "gamma_occupations.csv",
        np.column_stack((np.arange(len(occ)), occ)),
        delimiter=",",
        header="index,occupation",
        comments="",
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(
        min(-0.1, float(occ.min()) - 0.02), max(1.1, float(occ.max()) + 0.02), 49
    )
    ax.hist(occ, bins=bins, color="#2563a6", edgecolor="white")
    ax.axvline(0, color="#555555", lw=1, ls="--")
    ax.axvline(1, color="#555555", lw=1, ls="--")
    ax.set(
        xlabel="Occupation n (per spin)",
        ylabel="Number of states",
        title=(
            "Ground-truth density spectrum at Γ — Si validation snapshot 090"
            if args.case_index == 1
            else case["label"]
        ),
    )
    ax.text(
        0.5,
        0.95,
        f"{len(occ)} states · Σn = {occ.sum():.6f}\nmin = {occ.min():.5f} · max = {occ.max():.5f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    fig.savefig(out / "gamma_occupation_histogram.png", dpi=180)
    fig.savefig(out / "gamma_occupation_histogram.svg")
    plt.close(fig)
    inputs = [
        ROOT / case["matrix_path"],
        ROOT / case["info_path"],
        ROOT / Path(case["info_path"]).with_suffix(".cif"),
        *(
            pred_dir / f"pred_{name}.pt"
            for name in ("density", "overlap", "hamiltonian")
        ),
    ]
    payload = dict(
        case=case,
        nocc=nocc,
        mesh=args.mesh,
        mesh_convention="Uniform Gamma-centered fractional reciprocal grid arange(N)/N",
        occupation_convention="16 lowest occupied states per spin for 8 Si; zero-temperature CC^T; no fitted scaling",
        gamma_occupation_summary=dict(
            states=len(occ),
            minimum=float(occ.min()),
            maximum=float(occ.max()),
            sum=float(occ.sum()),
            below_zero=int((occ < 0).sum()),
            above_one=int((occ > 1).sum()),
            below_minus_001=int((occ < -0.01).sum()),
            above_101=int((occ > 1.01).sum()),
            ds_eigenproblem_residual=ds_residual,
            overlap_min_eigenvalue=float(se.min()),
        ),
        gamma=gamma_rows,
        sparse=sparse_rows,
        input_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
            if p.exists()
        },
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Hamiltonian-derived density and Γ occupations",
        "",
        case["label"],
        "",
        "Reference OpenMX density uses (D+Dᵀ)/2; legacy direct predictions are multiplied once by 0.5. All occupations and matrix errors use per-spin density.",
        "",
        "## Ground-truth Γ spectrum",
        "",
        "Γ matrices sum all stored periodic blocks. Occupations are eigenvalues of S^(1/2) D S^(1/2), which is similar to DS. No clipping is applied.",
        "",
        f"{len(occ)} states; min {occ.min():.8g}, max {occ.max():.8g}, sum {occ.sum():.8g}; {(occ<0).sum()} below zero and {(occ>1).sum()} above one. These are eigenvalues of the supplied finite-support density, not an untruncated Bloch density.",
        "",
        "![Gamma occupation histogram](gamma_occupation_histogram.png)",
        "",
        "## Γ density comparison",
        "",
        "Solve HC=SC E, normalize CᵀSC=I, and form D=C_occ C_occᵀ from the lowest 16 states (32 electrons). Using all eigenvectors would give S⁻¹. This is distinct from diagonalizing HS and treating its eigenvectors as AO coefficients.",
        "",
        "| Method | MAE | RMSE | Relative Frobenius |",
        "|---|---:|---:|---:|",
    ]
    for r in gamma_rows:
        lines.append(
            f"| {r['method']} | {r['mae']:.8g} | {r['rmse']:.8g} | {r['relative_frobenius']:.8g} |"
        )
    lines += [
        "",
        "## Periodic sparse density comparison",
        "",
        f"Reconstruct D(k)=C_occ(k) C_occ(k)† on a uniform Γ-centered {args.mesh}³ mesh, then inverse Fourier transform onto exactly the reference edges. The k mesh is explicitly chosen here; it is not assumed to reproduce OpenMX mesh offsets. The original SCF used an 8³ mesh at 300 K; this CC† reconstruction uses integer occupations. The reference H/S control measures the combined effect of reconstruction, finite support, mesh and occupation conventions.",
        "",
        "MAE/RMSE use all scalar entries over the union of periodic supports, matching the preceding McWeeny study. Band-energy errors use 2 Tr(D H_ref), eV/cell. Γ errors above and sparse errors below are different metrics.",
        "",
        "| Method | MAE | RMSE | Band-energy error (eV/cell) | Electron count | Global gap (eV) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sparse_rows:
        lines.append(
            f"| {r['method']} | {r['mae']:.8g} | {r['rmse']:.8g} | {r['band_energy_ref_h_abs_error_ev']:.8g} | {r['electrons']:.8g} | {r.get('global_gap_ev',float('nan')):.6g} |"
        )
    lines += [
        "",
        "All generalized-eigenproblem residuals and S-orthogonality errors are checked below 1e-10. Per-k occupation traces are checked against 16; inverse Fourier imaginary residuals are checked below 1e-10. Truncation can change the final sparse trace.",
        "",
        "Reproduce from the repository root:",
        "",
        "```bash",
        f"OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mandala-venv/bin/python scripts/evaluate_gamma_density.py --case-index {args.case_index} --mesh {args.mesh} --output-dir studies/mcweeny/gamma_density",
        "```",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                occupations=payload["gamma_occupation_summary"],
                gamma=gamma_rows,
                sparse=sparse_rows,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
