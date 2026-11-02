#!/usr/bin/env python3
"""Check the actual offline JavaScript against NumPy and MANDALA.

Run from the repository root: PYTHONPATH=src:. python demos/soc_rotations/verify.py
Node.js is needed only for this development check, never to open the app.
"""

import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from core.operator_layout import OrbitalFrameRotation


def array(value):
    return np.array([[complex(x["re"], x["im"]) for x in row] for row in value])


def main():
    torch.set_default_dtype(torch.float64)
    path = Path(__file__).with_name("index.html")
    rng = np.random.default_rng(8837)
    cases = []
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], *rng.normal(size=(5, 3))):
        for angle in (0.0, np.pi / 2, np.pi, 2 * np.pi, 4 * np.pi, 0.743):
            for inversion in (False, True):
                cases.append(
                    dict(
                        axis=np.asarray(axis).tolist(),
                        angle=angle,
                        inversion=inversion,
                        beta=1.13,
                        phi=0.49,
                        **dict(zip(("lambda", "delta", "field"), rng.uniform(0, 1, 3))),
                    )
                )
    program = r"""
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
vm.runInThisContext(html.match(/<script id="soc-math">([\s\S]*?)<\/script>/)[1]);
new vm.Script(html.match(/<script id="soc-interface">([\s\S]*?)<\/script>/)[1]);
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(s => SOCMath.calculate(s))));
"""
    outputs = json.loads(
        subprocess.run(
            ["node", "-e", program, str(path)],
            input=json.dumps(cases),
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    rotation = OrbitalFrameRotation(spinful=True, dtype=torch.float64)
    for state, result in zip(cases, outputs):
        h, hp = array(result["H"]), array(result["Hp"])
        d, u = array(result["D"]), array(result["U"])
        q = np.array(result["Q"])
        np.testing.assert_allclose(d.conj().T @ d, np.eye(6), atol=2e-14)
        np.testing.assert_allclose(q.T @ q, np.eye(3), atol=2e-14)
        np.testing.assert_allclose(
            np.linalg.det(q), -1 if state["inversion"] else 1, atol=2e-14
        )
        np.testing.assert_allclose(hp, d.conj().T @ h @ d, atol=2e-14)
        expected = rotation.rotate_blocks(
            torch.tensor(h), torch.tensor(q, dtype=torch.float64), [1], [1]
        ).numpy()
        np.testing.assert_allclose(hp, expected, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(
            result["energies"], np.linalg.eigvalsh(h), atol=3e-12
        )
        np.testing.assert_allclose(
            result["energiesPrime"], np.linalg.eigvalsh(hp), atol=3e-12
        )
        np.testing.assert_allclose(
            result["energiesPrime"], result["energies"], atol=3e-12
        )
        np.testing.assert_allclose(result["socError"], 0, atol=3e-14)
        # Spin is axial: inversion must not change its SU(2) action.
        np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=2e-14)
        r = np.array(result["R"])
        np.testing.assert_allclose(
            result["blochPrime"], r.T @ result["bloch"], atol=3e-14
        )
        # Conserved expectation for the same state in a changed spin basis.
        psi = np.array([complex(v["re"], v["im"]) for v in result["psi"]])
        prime = np.array([complex(v["re"], v["im"]) for v in result["psiPrime"]])
        original = np.array([[1, 0.4], [0.4, -0.5]])
        np.testing.assert_allclose(
            prime.conj() @ array(result["correct"]) @ prime,
            psi.conj() @ original @ psi,
            atol=3e-14,
        )
    # Exact bug example: a real off-diagonal becomes purely imaginary.
    quarter_turn = next(
        r
        for s, r in zip(cases, outputs)
        if s["axis"] == [0, 0, 1] and s["angle"] == np.pi / 2 and not s["inversion"]
    )
    np.testing.assert_allclose(array(quarter_turn["correct"])[0, 1], 0.4j, atol=2e-15)
    np.testing.assert_allclose(
        array(quarter_turn["broken"]), np.array([[1, 0.4], [0.4, -0.5]]) / 2, atol=2e-15
    )
    # Angular momentum addition: p × spin-1/2 splits into j=1/2 and j=3/2.
    special = dict(cases[0], **{"lambda": 0.6, "delta": 0, "field": 0})
    pure = json.loads(
        subprocess.run(
            ["node", "-e", program, str(path)],
            input=json.dumps([special]),
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )[0]
    np.testing.assert_allclose(
        pure["energies"], [-0.6, -0.6, 0.3, 0.3, 0.3, 0.3], atol=3e-13
    )
    print(
        f"PASS: {len(cases)} spin/orbital cases against MANDALA + NumPy; exact real-to-complex and L·S spectra."
    )


if __name__ == "__main__":
    main()
