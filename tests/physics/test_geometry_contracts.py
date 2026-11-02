"""Geometric representation contracts exercised from uncached raw graphs.

Periodic predictions are compared using directed (image, source, destination)
records, never by summing periodic images or comparing reordered flat tensors.
The explicitly named source-tie and hard-cutoff characterizations record known
limitations; they do not certify universal symmetry or continuity guarantees.
"""

from __future__ import annotations

import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from contract_support import tiny_model as build_model
from data.bond_geometry import bond_frames

KINDS = (
    "deeph",
    "deeph_e3",
    "mace_h",
    "tensor_messages_projected_attention",
    "tensor_products_degree_expansion_shell_readout",
    "equivariant_nodes_bond_frame_readout",
    "symmetric_products_attention_pair_readout",
)
SHELLS = {"H": [0], "O": [0, 1]}
SPECIES = ("H", "O", "H", "O")


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def geometry():
    return (
        torch.tensor(
            [
                [0.11, 0.07, 0.13],
                [1.24, 0.29, 0.27],
                [0.46, 1.51, 0.41],
                [0.34, 0.52, 1.69],
            ]
        ),
        torch.tensor([[3.71, 0.13, 0.07], [0.19, 3.83, 0.11], [0.09, 0.21, 3.97]]),
    )


def raw_graph(positions, species, box, *, edge_index, edge_shift, species_shells):
    vocabulary = tuple(species_shells)
    types = torch.tensor([vocabulary.index(symbol) for symbol in species])
    pairs = types[edge_index[0]] * len(vocabulary) + types[edge_index[1]]
    return dict(
        positions=positions,
        box=box,
        edge_index=edge_index,
        edge_shift=edge_shift,
        node_type_idx=types,
        node_one_hot=torch.nn.functional.one_hot(types, len(vocabulary)).to(positions),
        edge_type_idx=pairs,
        edge_one_hot=torch.nn.functional.one_hot(pairs, len(vocabulary) ** 2).to(
            positions
        ),
        node_batch=torch.zeros(len(species), dtype=torch.long),
        edge_batch=torch.zeros(edge_index.shape[1], dtype=torch.long),
    )


def graph(positions, box, species=SPECIES, cutoff=2.65):
    # ASE supplies only independently selected integer support. E3GNN must
    # derive vectors, lengths, harmonic features and local frames itself.
    a, b, images = neighbor_list(
        "ijS",
        Atoms(
            species,
            positions=positions.detach().numpy(),
            cell=box.detach().numpy(),
            pbc=True,
        ),
        cutoff,
        self_interaction=True,
    )
    return raw_graph(
        positions,
        species,
        box,
        edge_index=torch.tensor([a.tolist(), b.tolist()], dtype=torch.long),
        edge_shift=torch.tensor(images.T.copy(), dtype=torch.long),
        species_shells=SHELLS,
    )


def blocks(model, inputs):
    prediction = model(inputs, physical_derivatives=True)["hamiltonian"].to_blocks(
        model.mapper
    )
    return {
        key: prediction.pair_blocks[pair][row]
        for key, (pair, row) in prediction.lookup.items()
    }


def objective(model, inputs):
    # Frobenius norm is invariant under orbital frame rotations and touches
    # onsite, rectangular heterospecies and both directed offsite blocks.
    return sum(value.abs().square().sum() for value in blocks(model, inputs).values())


@pytest.mark.parametrize("kind", KINDS)
def test_atom_and_directed_edge_reordering_preserves_blocks_and_coordinate_gradients(
    kind,
):
    torch.manual_seed(54)
    model = build_model(kind, species_shells=SHELLS).eval()
    positions, box = geometry()
    positions.requires_grad_()
    original = graph(positions, box)
    baseline = blocks(model, original)
    parameters = tuple(model.parameters())
    baseline_loss = sum(value.abs().square().sum() for value in baseline.values())
    baseline_gradients = torch.autograd.grad(
        baseline_loss, (positions, *parameters), allow_unused=True
    )
    baseline_gradient = baseline_gradients[0]
    assert baseline_gradient.norm() > 1e-9
    assert any(key[3] != key[4] and baseline[key].shape == (1, 4) for key in baseline)

    # Rebuild raw metadata after relabeling, and independently shuffle the
    # directed records, including onsite records, across species partitions.
    order = torch.tensor([3, 0, 2, 1])  # new -> old
    changed_positions = positions.detach()[order].requires_grad_()
    changed = graph(changed_positions, box, tuple(SPECIES[i] for i in order))
    shuffle = torch.randperm(changed["edge_index"].shape[1])
    changed = raw_graph(
        changed_positions,
        tuple(SPECIES[i] for i in order),
        box,
        edge_index=changed["edge_index"][:, shuffle],
        edge_shift=changed["edge_shift"][:, shuffle],
        species_shells=SHELLS,
    )
    actual = blocks(model, changed)
    assert len(actual) == len(baseline)
    for key, value in actual.items():
        mapped = (*key[:3], int(order[key[3]]), int(order[key[4]]))
        torch.testing.assert_close(value, baseline[mapped], atol=2e-9, rtol=2e-8)
    changed_loss = sum(value.abs().square().sum() for value in actual.values())
    torch.testing.assert_close(changed_loss, baseline_loss, atol=2e-9, rtol=2e-8)
    changed_gradients = torch.autograd.grad(
        changed_loss, (changed_positions, *parameters), allow_unused=True
    )
    torch.testing.assert_close(
        changed_gradients[0], baseline_gradient[order], atol=2e-8, rtol=2e-6
    )
    for original_gradient, changed_gradient in zip(
        baseline_gradients[1:], changed_gradients[1:]
    ):
        if original_gradient is None:
            assert changed_gradient is None
        else:
            torch.testing.assert_close(
                changed_gradient, original_gradient, atol=2e-8, rtol=2e-6
            )


@pytest.mark.parametrize("kind", KINDS)
def test_periodic_atom_images_and_unimodular_lattice_basis_preserve_physical_blocks(
    kind,
):
    torch.manual_seed(55)
    model = build_model(kind, species_shells=SHELLS).eval()
    positions, box = geometry()
    original = graph(positions, box)
    expected = blocks(model, original)
    translations = torch.tensor([[1, -1, 0], [0, 1, 1], [-1, 0, 1], [1, 0, -1]])
    # Integer unimodular shear changes cell coordinates without changing the
    # lattice. New shifts satisfy L' U = L + t_i - t_j.
    shear = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    inverse = torch.linalg.inv(shear.double()).round().long()
    changed = graph(positions + translations.double() @ box, shear.double() @ box)
    assert changed["edge_shift"].abs().max() >= 2
    actual = blocks(model, changed)
    assert len(actual) == len(expected)
    for key, value in actual.items():
        i, j = key[3:]
        old_shift = torch.tensor(key[:3]) @ shear - translations[i] + translations[j]
        mapped = (*old_shift.tolist(), i, j)
        torch.testing.assert_close(value, expected[mapped], atol=2e-9, rtol=2e-8)
    # Ensure this exercises a genuine basis change, not just position wrapping.
    original_shift = original["edge_shift"].T
    torch.testing.assert_close((original_shift @ inverse) @ shear, original_shift)


@pytest.mark.parametrize("kind", KINDS)
def test_replicated_supercell_matches_every_primitive_directed_block(kind):
    torch.manual_seed(56)
    model = build_model(kind, species_shells=SHELLS).eval()
    positions, box = geometry()
    expected = blocks(model, graph(positions, box))
    super_positions = torch.cat([positions, positions + box[0]])
    super_box = box.clone()
    super_box[0] *= 2
    actual = blocks(model, graph(super_positions, super_box, SPECIES * 2))
    assert len(actual) == 2 * len(expected)
    mapped_counts = {key: 0 for key in expected}
    for key, value in actual.items():
        sx, sy, sz, i, j = key
        primitive = (2 * sx + j // 4 - i // 4, sy, sz, i % 4, j % 4)
        mapped_counts[primitive] += 1
        torch.testing.assert_close(value, expected[primitive], atol=3e-9, rtol=3e-8)
    assert set(mapped_counts.values()) == {2}
    # The invariant operator norm is extensive under a mere replication.
    torch.testing.assert_close(
        sum(v.square().sum() for v in actual.values()),
        2 * sum(v.square().sum() for v in expected.values()),
        atol=2e-9,
        rtol=2e-8,
    )


@pytest.mark.parametrize("kind", ["deeph", "equivariant_nodes_bond_frame_readout"])
def test_nearly_collinear_local_frames_have_correct_finite_derivatives_and_force_training_gradients(
    kind,
):
    torch.manual_seed(57)
    model = build_model(kind, species_shells=SHELLS).eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.35, 2e-4, 1e-4], [0.23, 1.9, 0.7]],
        requires_grad=True,
    )
    box = torch.eye(3) * 10
    index = torch.tensor([(i, j) for i in range(4) for j in range(4)]).T

    def at(value):
        return raw_graph(
            value,
            SPECIES,
            box,
            edge_index=index,
            edge_shift=torch.zeros(3, 16, dtype=torch.long),
            species_shells=SHELLS,
        )

    energy = objective(model, at(positions))
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    assert torch.isfinite(force).all() and force.norm() > 1e-7
    # This transverse component changes the near-singular frame, rather than
    # probing an unrelated radial coordinate in an otherwise easy geometry.
    step = 2e-8
    delta = torch.zeros_like(positions)
    delta[2, 2] = step
    finite_difference = -(
        objective(model, at(positions.detach() + delta))
        - objective(model, at(positions.detach() - delta))
    ) / (2 * step)
    torch.testing.assert_close(force[2, 2], finite_difference, atol=2e-6, rtol=3e-4)
    mixed = torch.autograd.grad(
        force.square().sum(), tuple(model.parameters()), allow_unused=True
    )
    active = [gradient for gradient in mixed if gradient is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert sum(gradient.square().sum() for gradient in active) > 1e-12


def test_exact_collinear_frame_is_rejected_and_equal_distance_gauge_tie_is_explicit():
    # There is no unique oriented bond frame in a collinear environment.
    index = torch.tensor([[0, 0, 0], [0, 1, 2]])
    with pytest.raises(ValueError, match="linearly independent"):
        bond_frames(
            index, torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        )
    # The source-faithful local frame uses record order to break a distance
    # tie. Document that limitation, rather than claiming false gauge smoothness.
    vectors = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    frames = bond_frames(index, vectors)
    swapped = bond_frames(index, vectors[[0, 2, 1]])
    assert not torch.allclose(frames[0], swapped[0])
    torch.testing.assert_close(frames[0].T @ frames[0], torch.eye(3))
    torch.testing.assert_close(torch.linalg.det(frames[0]), torch.tensor(1.0))


def test_zero_bond_harmonic_convention_is_identical_with_gradients_and_signed_zeros():
    import math
    from data.bond_geometry import tesseral_harmonics

    # Analytic real Y_l0 at the source's north-pole onsite convention.
    expected = torch.tensor(
        [
            math.sqrt(1 / (4 * math.pi)),
            0.0,
            math.sqrt(3 / (4 * math.pi)),
            0.0,
            0.0,
            0.0,
            math.sqrt(5 / (4 * math.pi)),
            0.0,
            0.0,
        ]
    )
    for sign in (-1.0, 1.0):
        for requires_grad in (False, True):
            zero = (sign * torch.zeros(1, 3)).requires_grad_(requires_grad)
            actual = tesseral_harmonics(zero, 3)
            torch.testing.assert_close(actual[0], expected, rtol=1e-14, atol=1e-14)


def test_tied_local_frame_limitation_is_visible_in_physical_model_predictions():
    """Characterize the unchanged DeepH source gauge, not a desired guarantee."""
    model = build_model("deeph", species_shells=SHELLS).eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.4, 1.7]]
    )
    index = torch.tensor([(i, j) for i in range(4) for j in range(4)]).T
    first = raw_graph(
        positions,
        SPECIES,
        torch.eye(3) * 10,
        edge_index=index,
        edge_shift=torch.zeros(3, 16, dtype=torch.long),
        species_shells=SHELLS,
    )
    order = torch.tensor([0, 2, 1, *range(3, 16)])
    second = raw_graph(
        positions,
        SPECIES,
        torch.eye(3) * 10,
        edge_index=index[:, order],
        edge_shift=torch.zeros(3, 16, dtype=torch.long),
        species_shells=SHELLS,
    )
    a, b = blocks(model, first), blocks(model, second)
    assert a.keys() == b.keys()
    # This explicitly records a source-faithful local model limitation. No
    # coordinate-axis or atom-index tiebreak can repair gauge ambiguity while
    # retaining all symmetry guarantees for every symmetric environment.
    assert (a[(0, 0, 0, 0, 0)] - b[(0, 0, 0, 0, 0)]).abs().max() > 1e-6


@pytest.mark.parametrize(
    "kind",
    [
        "deeph",
        "deeph_e3",
        "mace_h",
        "tensor_messages_projected_attention",
        "equivariant_nodes_bond_frame_readout",
    ],
)
def test_periodic_cell_strain_derivative_matches_finite_difference_and_has_finite_second_derivative(
    kind,
):
    model = build_model(kind, species_shells=SHELLS).eval()
    positions, box = geometry()
    baseline = graph(positions, box)
    assert (baseline["edge_shift"] != 0).any()
    strain = torch.tensor(0.0, requires_grad=True)
    direction = torch.tensor([[0.3, 0.2, 0.0], [0.2, -0.1, 0.05], [0.0, 0.05, 0.15]])

    def energy(value):
        deformation = torch.eye(3) + value * direction
        return objective(
            model,
            {
                **baseline,
                "positions": positions @ deformation,
                "box": box @ deformation,
            },
        )

    value = energy(strain)
    derivative = torch.autograd.grad(value, strain, create_graph=True)[0]
    curvature = torch.autograd.grad(derivative, strain)[0]
    step = 2e-5
    plus, minus = energy(step), energy(-step)
    torch.testing.assert_close(
        derivative, (plus - minus) / (2 * step), atol=2e-7, rtol=3e-4
    )
    torch.testing.assert_close(
        curvature, (plus - 2 * value + minus) / step**2, atol=2e-8, rtol=3e-3
    )
    assert derivative.abs() > 1e-8 and torch.isfinite(curvature)


def cutoff_model(smooth):
    from core.block_irrep_mapper import BlockIrrepMapper
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from net.configuration import resolve_configuration
    from net.common import Config
    from net.e3gnn import E3GNN

    radial = dict(
        id="radial",
        transform="cutoff_radial_expansion" if smooth else "gaussian_basis",
        inputs=["message_lengths"],
        outputs=["radial"],
        options=(
            dict(r_max=2.0, num_basis=4)
            if smooth
            else dict(start=0.0, stop=2.0, n_gaussians=4)
        ),
    )
    specification = dict(
        schema_version=1,
        species=dict(ordering="atomic_number"),
        data=dict(representation="packed_shells", basis="openmx"),
        pipeline=dict(
            stages=[
                radial,
                dict(
                    id="sum",
                    transform="indexed_aggregation",
                    inputs=["radial", "message_edge_index", "node_one_hot"],
                    outputs=["node_sum"],
                    options=dict(reduction="sum", receiver=0),
                ),
                dict(
                    id="gather",
                    transform="endpoint_gather",
                    inputs=["node_sum", "edge_one_hot", "edge_index"],
                    outputs=["endpoints"],
                ),
                dict(
                    id="readout",
                    transform="irrep_linear",
                    inputs=["endpoints"],
                    outputs=["coupled"],
                    options=dict(irreps_out="${layout.coupled_irreps}"),
                ),
                dict(
                    id="decode",
                    transform="shell_tensor_decode",
                    inputs=["coupled"],
                    outputs=["matrix"],
                    options=dict(shell_pairs="${layout.shell_pairs}", spinful=False),
                ),
            ],
            outputs=dict(hamiltonian="matrix"),
        ),
    )
    resolved, _ = resolve_configuration(specification, species_shells=SHELLS)
    return (
        E3GNN(
            BlockIrrepMapper(
                OrbitalIrrepConfig.from_dict({"H": "1s", "O": "1s1p"}),
                dtype=torch.float64,
            ),
            Config.from_mapping(resolved),
        )
        .double()
        .eval()
    )


def test_rebuilt_cutoff_topology_is_smooth_only_when_the_selected_message_map_vanishes():
    # Real configurable E3GNNs share aggregation/readout and differ only in the
    # selected radial map. No normalization/attention denominator silently
    # reintroduces dependence on a zero-valued disappearing edge.
    for smooth in (True, False):
        torch.manual_seed(60)
        model = cutoff_model(smooth)

        def onsite(distance):
            positions = torch.stack(
                [
                    torch.zeros(3),
                    torch.stack([distance, distance * 0.0, distance * 0.0]),
                ]
            )
            current = graph(
                positions, torch.eye(3) * 10, species=("H", "O"), cutoff=2.0
            )
            value = blocks(model, current)[(0, 0, 0, 0, 0)].sum()
            return value, current

        outside, absent = onsite(torch.tensor(2.01))
        errors, slopes = [], []
        for separation in (0.1, 0.05, 0.025):
            distance = torch.tensor(2.0 - separation, requires_grad=True)
            inside, present = onsite(distance)
            assert present["edge_index"].shape[1] == absent["edge_index"].shape[1] + 2
            errors.append((inside - outside).abs())
            slopes.append(torch.autograd.grad(inside, distance)[0].abs())
        assert all(torch.isfinite(value) for value in errors + slopes)
        if smooth:
            assert errors[0] > 1e-8
            assert errors[2] < errors[0] / 100
            assert slopes[2] < slopes[0] / 25
        else:
            # Gaussian expansion has no compact support: merely deleting a
            # neighbor therefore produces a finite operator discontinuity.
            boundary_a = (onsite(torch.tensor(1.999))[0] - outside).abs()
            boundary_b = (onsite(torch.tensor(1.9995))[0] - outside).abs()
            assert boundary_b > 1e-5
            torch.testing.assert_close(boundary_a, boundary_b, rtol=0.02, atol=1e-7)
