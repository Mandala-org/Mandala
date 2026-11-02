# SOC rotation lab

Open `index.html` directly in a modern browser. The single file contains all
equations, graphics and calculations; no server, installation or network is
needed. The page also blocks network connections through its content policy.

- Drag the orbital image; select a p orbital, rotation axis, angle or inversion.
- Compare a fixed lab spin with its components in the rotated basis.
- Inspect the complex 6×6 Hamiltonian and its invariant spectrum.
- Select **Pure λ L·S** for the j=1/2 doublet and j=3/2 quartet.
- Compare the correct complex rotation with the old real-casting failure.

The convention is a **passive basis change**: `ψ′ = D†ψ`, `H′ = D†HD`,
`D = U ⊗ Q`, spin-major Cartesian p ordering. The SU(2) lift is continuous in
the angle, including its sign change at 2π. For an improper spatial rotation,
spin uses the proper axial rotation `det(Q) Q`. Orbital lobe images show angular
amplitude, with a schematic radial envelope; the Hamiltonian is illustrative.

Development verification, from the repository root:

```bash
PYTHONPATH=src:. python demos/soc_rotations/verify.py
```

This executes the actual JavaScript mathematics with Node.js and compares it
against NumPy eigensolvers and MANDALA native SOC frame rotations. Node.js and
the Python environment are only needed for verification, not for using the app.
