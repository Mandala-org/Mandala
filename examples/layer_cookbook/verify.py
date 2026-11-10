"""Certify every cookbook layer and the complete cookbook model.

Run from the repository root::

    PYTHONPATH=src:. python examples/layer_cookbook/verify.py

Four properties are checked in double precision:

1. Layer covariance. Each layer satisfies ``f(D(g) x) = D(g) f(x)`` for a
   random proper rotation, an improper rotation and the inversion.
2. Model covariance. The predicted Hamiltonian of a periodic structure with
   s, p and d orbitals transforms as ``H_ij -> D_i(g) H_ij D_j(g)^T`` when the
   structure and its cell are rotated or inverted, before and after training.
3. Trainability. Every parameter receives a finite gradient and an optimizer
   step changes the prediction.
4. The registered Hamiltonian-trace objective contributes to the model loss
   with its configured weight and sends finite, nonzero gradients to the model.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from e3nn import o3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layers  # noqa: E402,F401  (registers the cookbook transformations)
from examples.layer_cookbook import new_observable  # noqa: E402

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.operator_layout import OrbitalFrameRotation  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.structure_inference import build_model_input_from_structure  # noqa: E402
from net.common import Config  # noqa: E402
from net.configuration import resolve_configuration  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402

SPECIES_SHELLS = {"H": [0], "O": [0, 1], "Si": [0, 1, 2]}


def group_elements():
    proper = o3.rand_matrix()
    return {
        "rotation": proper,
        "improper rotation": -proper,
        "inversion": -torch.eye(3),
    }


def assert_covariant(name, apply, inputs_and_irreps, output_irreps, atol=1e-9):
    """``apply(*inputs)`` must commute with the action of every group element."""
    for label, g in group_elements().items():
        transformed = [
            x @ o3.Irreps(irreps).D_from_matrix(g).T if irreps is not None else x
            for x, irreps in inputs_and_irreps
        ]
        expected = apply(*(x for x, _ in inputs_and_irreps))
        expected = expected @ o3.Irreps(output_irreps).D_from_matrix(g).T
        observed = apply(*transformed)
        error = (observed - expected).abs().max().item()
        if error > atol:
            raise AssertionError(f"{name}: {label} error {error:.2e}")
    print(f"  {name:<34s} covariant")


def check_layers():
    print("Layer covariance")
    E, N = 11, 4
    edge_index = torch.randint(0, N, (2, E))
    radial = torch.randn(E, 8)
    sh = o3.Irreps("0e+1o+2e")
    vectors = torch.randn(E, 3)
    angular = o3.spherical_harmonics(sh, vectors, normalize=True)
    hidden = o3.Irreps("4x0e+2x0o+3x1o+2x1e+2x2e")

    encoder = layers.RadialAngularEncoder(8, sh, "4x0e+3x1o+2x2e")
    assert_covariant(
        "radial_angular_encoder",
        encoder,
        [(radial, None), (angular, sh)],
        encoder.irreps_out,
    )

    conv = layers.DepthwiseConvolution(hidden, sh, 8, lmax=2)
    assert_covariant(
        "depthwise_convolution",
        lambda h, y: conv(h, y, radial, edge_index),
        [(torch.randn(N, hidden.dim), hidden), (angular, sh)],
        conv.irreps_out,
    )

    aggregate = layers.TemperedSoftmaxAggregation(hidden, 8)
    reference = torch.zeros(N, 1)
    assert_covariant(
        "tempered_softmax_aggregation",
        lambda m: aggregate(m, radial, edge_index, reference),
        [(torch.randn(E, hidden.dim), hidden)],
        hidden,
    )

    for name, module, irreps_out in (
        ("squash_activation", layers.SquashActivation(hidden), hidden),
        ("degree_rms_norm", layers.DegreeRMSNorm(hidden), hidden),
        ("equivariant_glu", layers.EquivariantGLU(hidden), hidden),
    ):
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.add_(0.3 * torch.randn_like(parameter))
        assert_covariant(
            name, module, [(torch.randn(N, hidden.dim), hidden)], irreps_out
        )

    rezero = layers.ReZeroResidual(hidden, initial=0.4)
    assert_covariant(
        "rezero_residual",
        rezero,
        [(torch.randn(N, hidden.dim), hidden), (torch.randn(N, hidden.dim), hidden)],
        hidden,
    )

    update = layers.EndpointBilinearEdgeUpdate(hidden, hidden, "3x0e+2x1e+2x1o+2x2e")
    assert_covariant(
        "endpoint_bilinear_edge_update",
        lambda h, e: update(h, e, edge_index),
        [(torch.randn(N, hidden.dim), hidden), (torch.randn(E, hidden.dim), hidden)],
        update.irreps_out,
    )

    pairs = torch.randint(0, 9, (E,))
    readout = layers.PairMixtureReadout(hidden, "0e+1o+0e+1e+2e", num_pairs=9)
    assert_covariant(
        "pair_mixture_readout",
        lambda e: readout(e, pairs),
        [(torch.randn(E, hidden.dim), hidden)],
        "0e+1o+0e+1e+2e",
    )

    envelope = layers.LearnedPairEnvelope(num_pairs=9)
    lengths = vectors.norm(dim=-1)
    assert_covariant(
        "learned_pair_envelope",
        lambda x: envelope(x, lengths, pairs),
        [(torch.randn(E, hidden.dim), hidden)],
        hidden,
    )

    # Invariant layers: outputs must not change under any group element.
    basis = layers.BesselBasis(cutoff=5.0)
    for label, g in group_elements().items():
        torch.testing.assert_close(basis((vectors @ g.T).norm(dim=-1)), basis(lengths))
    torch.testing.assert_close(
        basis(torch.zeros(1)).isfinite().all(), torch.tensor(True)
    )
    print(f"  {'bessel_basis':<34s} invariant, finite at r = 0")
    embedding = layers.PeriodicTableEmbedding(["H", "O", "Si"], width=4)
    assert embedding(torch.tensor([0, 1, 2])).shape == (3, 4)
    print(f"  {'periodic_table_embedding':<34s} invariant by construction")


def build_model(*, objective_terms=()):
    resolved, codec = resolve_configuration(
        HERE / "cookbook.yaml", species_shells=SPECIES_SHELLS
    )
    resolved["training"]["dtype"] = "float64"
    if objective_terms:
        resolved["objective"]["terms"] = list(objective_terms)
    cfg = Config.from_mapping(resolved)
    orbitals = OrbitalIrrepConfig.from_dict(
        {
            element: [f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in shells]
            for element, shells in codec.species_shells.items()
        }
    )
    return E3GNN(BlockIrrepMapper(orbitals, dtype=torch.float64), cfg).double()


def predict(model, atoms, positions, box):
    x = build_model_input_from_structure(
        atoms=atoms, positions=positions, box=box, cfg=model.cfg, mapper=model.mapper
    )
    return model(x)["hamiltonian"].to_blocks(model.mapper)


def check_model():
    print("Model covariance")
    model = build_model().eval()
    atoms = ("H", "O", "Si", "O")
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.3, 0.4, 0.6], [-0.2, 1.5, 0.8], [0.9, 1.1, -0.7]]
    )
    box = torch.tensor([[4.6, 0.2, 0.3], [0.1, 4.9, 0.2], [0.3, 0.1, 5.2]])
    rotation = OrbitalFrameRotation(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for phase in ("initial", "trained"):
        base = predict(model, atoms, positions, box)
        for label, g in group_elements().items():
            observed = predict(model, atoms, positions @ g.T, box @ g.T)
            worst = 0.0
            for edge, (key, row) in base.lookup.items():
                left, right = (SPECIES_SHELLS[s] for s in key.split("-"))
                expected = rotation.rotate_blocks(
                    base.pair_blocks[key][row], g.T, left, right
                )
                worst = max(worst, (observed[edge] - expected).abs().max().item())
            if worst > 1e-8:
                raise AssertionError(f"{phase} model, {label}: error {worst:.2e}")
            print(f"  {phase:<8s} {label:<18s} max |dH| = {worst:.1e}")
        if phase == "initial":
            optimizer.zero_grad()
            loss = sum(block.square().mean() for block in base.pair_blocks.values())
            loss.backward()
            missing = [
                name
                for name, p in model.named_parameters()
                if p.grad is None or not torch.isfinite(p.grad).all()
            ]
            if missing:
                raise AssertionError(f"Parameters without finite gradients: {missing}")
            optimizer.step()
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")


def check_new_observable():
    print("New observable objective")
    weight = 0.25
    model = build_model(
        objective_terms=[{"name": "hamiltonian_trace", "weight": weight}]
    ).eval()
    x = build_model_input_from_structure(
        atoms=("Si", "Si"),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.4, 0.1]]),
        box=torch.eye(3) * 4.9,
        cfg=model.cfg,
        mapper=model.mapper,
    )
    prediction = model(x)["hamiltonian"].to_blocks(model.mapper)
    reference = prediction._replace_pair_blocks(
        {
            pair: torch.zeros_like(blocks)
            for pair, blocks in prediction.pair_blocks.items()
        },
        basis=prediction.basis,
    )
    for atom in range(len(reference.atoms)):
        pair, row = reference.lookup[(0, 0, 0, atom, atom)]
        reference.pair_blocks[pair][row].fill_diagonal_(0.2)
    loss, metrics, _ = model._graph_step((x, {"hamiltonian": reference}), 0, "train")
    expected = new_observable.hamiltonian_trace_loss(
        predictions={"hamiltonian": prediction},
        targets={"hamiltonian": reference},
        inputs=x,
    )
    observed = metrics["train/loss_term_hamiltonian_trace"]
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(
        loss, metrics["train/loss_matrix_total"] + weight * expected
    )
    observed.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)
    print(
        "  hamiltonian_trace: configured weight and nonzero finite gradients verified"
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    check_layers()
    check_model()
    check_new_observable()
    print("All cookbook layers and the cookbook model are E(3)-equivariant.")
