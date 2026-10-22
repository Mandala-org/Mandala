"""Actual E3GNN training, train-only statistics and segmented continuation."""

from pathlib import Path
import json
import hashlib
import shutil

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from scripts.train import main

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_factory_consumes_openmx_without_conversion(tmp_path, name):
    from data.deeph_e3_exporter import export_openmx_snapshot_to_deeph_e3
    from data.gnn_dataset import E3GNNDataset
    from scripts.train import build_components

    raw, native = {}, {}
    for split in ("train", "val"):
        directory = tmp_path / "raw" / split
        directory.mkdir(parents=True)
        for filename in ("HS.out", "Si.out", "Si.cif"):
            shutil.copyfile(ROOT / "data/small/Si" / filename, directory / filename)
        destination = tmp_path / "native" / split
        export_openmx_snapshot_to_deeph_e3(
            directory / "HS.out", directory / "Si.out", destination
        )
        raw[split], native[split] = [str(directory)], [str(destination)]
    raw["test"], native["test"] = [], []
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        raw_config = OmegaConf.load(document(tmp_path, name, raw))
        if name != "deeph":
            raw_config.parameters.angular_irreps = "1x0e+1x1o+1x2e+1x3o+1x4e"
        _, _, raw_model, raw_ds, _ = build_components(raw_config)
        native_config = OmegaConf.merge(raw_config, {"data": {"splits": native}})
        _, _, native_model, native_ds, _ = build_components(native_config)
        assert isinstance(raw_ds["train"], E3GNNDataset)
        assert isinstance(native_ds["train"], E3GNNDataset)
        x_raw, y_raw = raw_ds["train"][0]
        x_native, y_native = native_ds["train"][0]
        torch.testing.assert_close(
            y_raw["hamiltonian"]["values"],
            y_native["hamiltonian"]["values"],
            rtol=1e-10,
            atol=1e-10,
        )
        for key in ("edge_vectors", "edge_index", "edge_type_idx"):
            torch.testing.assert_close(
                x_raw[key], x_native[key], rtol=1e-10, atol=1e-10
            )
        raw_model.eval()
        native_model.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                raw_model(x_raw)["hamiltonian"].packed_values,
                native_model(x_native)["hamiltonian"].packed_values,
                rtol=1e-9,
                atol=1e-9,
            )
    finally:
        torch.set_default_dtype(previous)


def native_data(root, *, spinful=False):
    paths = []
    for sample in range(4):
        directory = root / str(sample)
        directory.mkdir(parents=True)
        np.savetxt(directory / "element.dat", [1, 1, 1], fmt="%d")
        np.savetxt(
            directory / "site_positions.dat",
            np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]]).T,
        )
        np.savetxt(directory / "lat.dat", np.eye(3) * 6)
        (directory / "orbital_types.dat").write_text("0\n0\n0\n")
        (directory / "info.json").write_text(json.dumps({"isspinful": spinful}))
        for filename in ("hamiltonians.h5", "overlaps.h5"):
            with h5py.File(directory / filename, "w") as handle:
                for i in range(3):
                    for j in range(3):
                        value = 0.1 * sample + 0.01 * (i + j)
                        if spinful:
                            if sample >= 2:
                                value += 10000.0  # Held-out outliers must not fit output statistics.
                            block = np.array(
                                [
                                    [value + 0.1, 0.02 + 0.03j],
                                    [0.04 - 0.01j, value - 0.1],
                                ],
                                dtype=np.complex128,
                            )
                            if filename == "hamiltonians.h5" and i == 0 and j == 1:
                                block[:] = np.nan + 1j * np.nan
                        else:
                            block = np.array([[value]])
                        handle[str([0, 0, 0, i + 1, j + 1])] = block
        paths.append(str(directory))
    return {"train": paths[:2], "val": paths[2:3], "test": paths[3:]}


def document(tmp_path, name, splits):
    config = OmegaConf.load(ROOT / f"configs/reference/{name}.yaml")
    if name == "deeph":
        parameters = {
            "node_width": 3,
            "edge_width": 7,
            "post_edge_width": 3,
            "num_l": 2,
            "angular_dim": 4,
        }
    else:
        parameters = {
            "embedding_width": 3,
            "radial_width": 3,
            "node_irreps": "3x0e+2x1o",
            "edge_irreps": "3x0e+2x1o",
            "angular_irreps": "1x0e+1x1o",
        }
        if name == "mace_h":
            parameters.update(
                hidden_irreps="3x0e+3x1o",
                density_lmax=1,
                density_radial_width=3,
                correlation=2,
                radial_layers=[4],
            )
    config = OmegaConf.merge(
        config,
        {
            "data": {"splits": splits},
            "parameters": parameters,
            "training": {"max_epochs": 2, "dtype": "float64", "batch_size": 2},
        },
    )
    path = tmp_path / f"{name}.yaml"
    OmegaConf.save(config, path, resolve=False)
    return path


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_three_reference_presets_through_actual_training_cli(tmp_path, name):
    splits = native_data(tmp_path / "data")
    config = document(tmp_path, name, splits)
    previous = torch.get_default_dtype()
    try:
        main(["--config", str(config), "--output", str(tmp_path / "dry"), "--dry-run"])
        if name == "mace_h":
            main(
                [
                    "--config",
                    str(config),
                    "--output",
                    str(tmp_path / "run"),
                    "--accelerator",
                    "cpu",
                    "--max-wall-clock-seconds",
                    "0.000001",
                    "--log-dir",
                    str(tmp_path / "metrics"),
                ]
            )
            checkpoint = tmp_path / "run/latest_checkpoint.pt"
            first = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert first["epoch_boundary_resume"]["safe_to_resume"]
            assert list((tmp_path / "metrics").rglob("metrics.csv"))
            specification = json.loads((tmp_path / "run/run_manifest.json").read_text())
            affine = specification["specification"]["statistics"]["output_affine"]
            assert set(affine["training_ids"]) == set(splits["train"])
            assert not set(affine["training_ids"]) & set(splits["val"] + splits["test"])
            assert (
                first["state_dict"]["pipeline.transforms.output_affine._extra_state"][
                    "training_manifest_hash"
                ]
                == affine["training_manifest_hash"]
            )
            main(
                [
                    "--config",
                    str(config),
                    "--output",
                    str(tmp_path / "run"),
                    "--accelerator",
                    "cpu",
                    "--resume",
                    str(checkpoint),
                ]
            )
            resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert resumed["epoch"] > first["epoch"]
            main(
                [
                    "--config",
                    str(config),
                    "--output",
                    str(tmp_path / "uninterrupted"),
                    "--accelerator",
                    "cpu",
                ]
            )
            uninterrupted = torch.load(
                tmp_path / "uninterrupted/latest_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert resumed["state_dict"].keys() == uninterrupted["state_dict"].keys()
            for key, value in resumed["state_dict"].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        value, uninterrupted["state_dict"][key], rtol=0, atol=0
                    )
                else:
                    assert value == uninterrupted["state_dict"][key]
    finally:
        torch.set_default_dtype(previous)


def test_split_ratios_do_not_assign_rounding_remainder_to_test(tmp_path):
    from scripts.train import dataset_paths

    splits = native_data(tmp_path / "data")
    root = Path(splits["train"][0]).parent
    document = {
        "data": {"root": str(root)},
        "training": {
            "split": {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.0}
        },
    }
    resolved = dataset_paths(document)
    assert {key: len(value) for key, value in resolved.items()} == {
        "train": 2,
        "val": 1,
        "test": 0,
    }
    document["training"]["split"]["test_ratio"] = 0.25
    assert len(dataset_paths(document)["test"]) == 1


@pytest.mark.parametrize(
    "options,match",
    [
        ({"max_epocs": 2}, "Unknown training"),
        ({"batch_size": 0}, "batch_size"),
        ({"split": {"test_ratios": 0.2}}, "Unknown split"),
        ({"split": {"test_ratio": -0.1}}, "test_ratio"),
        ({"split": {"test_size": 1.5}}, "test_size"),
    ],
)
def test_training_options_fail_before_loading_data(options, match):
    from scripts.train import dataset_paths

    with pytest.raises(ValueError, match=match):
        dataset_paths({"data": {"root": "/path/never/loaded"}, "training": options})


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_complex_soc_presets_native_dataset_through_lightning_fit(tmp_path, name):
    from scripts.train import build_components

    splits = native_data(tmp_path / "data", spinful=True)
    config_path = document(tmp_path, name, splits)
    config = OmegaConf.load(config_path)
    config.training.max_epochs = 1
    # Support is independent of the masked Hamiltonian entries.
    config.data.support_file = "overlaps.h5"
    OmegaConf.save(config, config_path, resolve=False)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        resolved, codec, model, datasets, manifest = build_components(config)
        assert codec.spinful
        batch = datasets["train"][0]
        record = batch[1]["hamiltonian"]
        assert record["values"].is_complex()
        assert not record["mask"].all()
        assert torch.isfinite(record["values"]).all()
        if name == "mace_h":
            affine = model.pipeline.transforms["output_affine"]
            assert affine.training_manifest_hash
            assert torch.isfinite(affine.std).all()
            assert affine.mean.abs().max() < 1.0
            assert affine.std.max() <= 1.0
            # Physical derivatives propagate through the actual configured
            # stream; reference training keeps the intentional surrogate.
            _, identity_trace = model.pipeline(batch[0], return_trace=True)
            _, physical_trace = model.pipeline(
                batch[0], return_trace=True, physical_derivatives=True
            )
            torch.testing.assert_close(
                identity_trace["coupled"], physical_trace["coupled"]
            )
            (surrogate,) = torch.autograd.grad(
                identity_trace["coupled"].square().sum(), identity_trace["coupled_raw"]
            )
            (physical,) = torch.autograd.grad(
                physical_trace["coupled"].square().sum(), physical_trace["coupled_raw"]
            )
            torch.testing.assert_close(
                physical, surrogate * affine.std[batch[0]["edge_type_idx"]]
            )
            assert not torch.allclose(physical, surrogate)
        main(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "dry"),
                "--dry-run",
            ]
        )
        main(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "run"),
                "--accelerator",
                "cpu",
            ]
        )
        checkpoint = torch.load(
            tmp_path / "run/latest_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["epoch_boundary_resume"]["safe_to_resume"]
        assert checkpoint["global_step"] == 1
        # The public prediction CLI must recover a real trained checkpoint and
        # build support from geometry without reading any reference matrices.
        from scripts.export_prediction import main as predict_main

        geometry = tmp_path / "geometry"
        geometry.mkdir()
        source = Path(splits["test"][0])
        for filename in (
            "element.dat",
            "site_positions.dat",
            "lat.dat",
            "orbital_types.dat",
            "info.json",
        ):
            shutil.copyfile(source / filename, geometry / filename)
        export = tmp_path / "predictions"
        checkpoint_path = tmp_path / "run/latest_checkpoint.pt"
        predict_main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--snapshot",
                str(geometry),
                "--format",
                "native",
                "--output",
                str(export),
                "--graph-radius",
                "2",
            ]
        )
        with (
            h5py.File(export / "hamiltonians.h5") as matrices,
            h5py.File(export / "hamiltonian_coverage.h5") as coverage,
        ):
            assert len(matrices) == 9
            assert set(matrices) == set(coverage)
            for key in matrices:
                assert matrices[key].shape == (2, 2)
                assert np.iscomplexobj(matrices[key][...])
                assert np.isfinite(matrices[key][...]).all()
                assert coverage[key].dtype == np.bool_
                assert coverage[key][...].all()
        provenance = json.loads((export / "info.json").read_text())["prediction"]
        assert provenance["frame"] == "global"
        assert provenance["orbital_basis"] == "openmx"
        assert provenance["hermitian_projection"] is False
        assert (
            provenance["sha256"]
            == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        )
        assert provenance["checkpoint"] == str(checkpoint_path.resolve())
        assert all(
            torch.isfinite(value).all()
            for value in checkpoint["state_dict"].values()
            if isinstance(value, torch.Tensor)
        )
    finally:
        torch.set_default_dtype(previous)


def test_radius_discovery_does_not_require_reference_matrices(tmp_path):
    from scripts.train import dataset_paths

    splits = native_data(tmp_path / "data")
    root = Path(splits["train"][0]).parent
    for path in root.rglob("*.h5"):
        path.unlink()
    document = {
        "data": {"root": str(root), "graph_source": "radius", "graph_radius": 2.0},
        "training": {"split": {"train_size": 2, "val_size": 1, "test_size": 1}},
    }
    assert {key: len(value) for key, value in dataset_paths(document).items()} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    document["data"]["graph_source"] = "stored"
    with pytest.raises(ValueError, match="No operator snapshots"):
        dataset_paths(document)


@pytest.mark.parametrize(
    "radius", [None, 0, -1.0, float("nan"), float("inf"), True, "2"]
)
def test_radius_training_configuration_requires_finite_positive_radius(radius):
    from scripts.train import dataset_paths

    with pytest.raises(ValueError, match="finite positive graph_radius"):
        dataset_paths(
            {
                "data": {
                    "root": "/not/accessed",
                    "graph_source": "radius",
                    "graph_radius": radius,
                }
            }
        )


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_reference_graphs_compose_with_envelope_observables_and_spectral_loss(
    tmp_path, name
):
    """Scientific objectives operate on the shared physical matrix contract."""
    from scripts.train import build_components
    from data.sample import collate_graph_samples

    splits = native_data(tmp_path / "data")
    for paths in splits.values():
        for path in paths:
            path = Path(path)
            (path / "info.json").write_text(
                json.dumps({"isspinful": False, "fermi_level": 0.0})
            )
            for filename in ("overlaps.h5", "density_matrixs.h5"):
                with h5py.File(path / filename, "w") as handle:
                    for i in range(3):
                        for j in range(3):
                            handle[str([0, 0, 0, i + 1, j + 1])] = np.array(
                                [[float(i == j)]]
                            )
    envelope = tmp_path / "envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "family": "slater_soft_cutoff",
                "pairs": {"H-H": {"theta": [0.0, -1.0, 0.0, 4.0, 0.0]}},
            }
        )
    )
    config = OmegaConf.load(document(tmp_path, name, splits))
    config.update(
        hamiltonian_envelope_path=str(envelope),
        hamiltonian_envelope_mode="multiply_prediction",
        train_on_energy=True,
        loss_coef_observables=0.2,
        spectral_loss_enabled=True,
        spectral_loss_coef=0.1,
        spectral_loss_kmesh="1x1x1",
        log_per_irrep_metrics=False,
        log_hamiltonian_irrep_contrib_metrics=False,
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, model, datasets, _ = build_components(config)
        logged = {}
        model.log_dict = lambda values, **kwargs: logged.update(values)
        batch = collate_graph_samples(datasets["train"].samples)
        loss = model._shared_step(batch, 0, "train")
        loss.backward()
        assert torch.isfinite(loss)
        assert logged["train/loss_energy_weighted"] > 0
        assert logged["train/loss_spectral_weighted"] >= 0
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in model.parameters()
        )
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        assert all(
            "spectral_eigvals_ref" in sample[0]
            or any("spectral" in key for key in sample[0])
            for sample in datasets["train"].samples
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_shared_batch_step_matches_disjoint_source_reduction_and_normalization(
    tmp_path, name
):
    from scripts.train import build_components
    from data.sample import collate_graph_samples, disjoint_union_samples
    from net.training_policies import masked_error

    splits = native_data(tmp_path / "data")
    config = OmegaConf.load(document(tmp_path, name, splits))
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, model, datasets, _ = build_components(config)
        samples = datasets["train"].samples
        # Unequal coverage checks reduction over entries rather than graph means.
        samples[1][1]["hamiltonian"]["mask"][::2] = False
        _, _, expected, _, _ = build_components(config)
        expected.load_state_dict(model.state_dict())
        union, targets = disjoint_union_samples(
            [
                {
                    "inputs": sample[0],
                    "targets": {"hamiltonian": sample[1]["hamiltonian"]},
                }
                for sample in samples
            ]
        )
        prediction = expected(union)["hamiltonian"].packed_values
        reference_loss = masked_error(
            prediction,
            targets["hamiltonian"]["values"],
            targets["hamiltonian"]["mask"],
            kind=config.objective.kind,
        )
        model.log_dict = lambda values, **kwargs: None
        loss = model._shared_step(collate_graph_samples(samples), 0, "train")
        torch.testing.assert_close(loss, reference_loss, rtol=1e-11, atol=1e-12)
        loss.backward()
        reference_loss.backward()
        for (name_a, param_a), (name_b, param_b) in zip(
            model.named_parameters(), expected.named_parameters()
        ):
            assert name_a == name_b
            if param_a.grad is not None:
                torch.testing.assert_close(
                    param_a.grad, param_b.grad, rtol=1e-9, atol=1e-10
                )
        for (name_a, buffer_a), (name_b, buffer_b) in zip(
            model.named_buffers(), expected.named_buffers()
        ):
            assert name_a == name_b
            torch.testing.assert_close(buffer_a, buffer_b)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_common_force_and_stress_paths_match_physical_energy_derivatives(
    tmp_path, name
):
    from scripts.train import build_components

    splits = native_data(tmp_path / "data")
    for paths in splits.values():
        for path in paths:
            for filename in ("hamiltonians.h5", "overlaps.h5"):
                with h5py.File(Path(path) / filename, "a") as handle:
                    handle[str([1, 0, 0, 1, 1])] = np.array([[0.03]])
                    handle[str([-1, 0, 0, 1, 1])] = np.array([[0.03]])
    config = OmegaConf.load(document(tmp_path, name, splits))
    # One shared graph can supply multiple operators without a model subclass.
    for operator in ("density", "overlap"):
        config.pipeline.outputs[operator] = config.pipeline.outputs.hamiltonian
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, model, datasets, _ = build_components(config)
        model.eval()
        model.cfg.train_on_forces = True
        model.cfg.train_on_stress = True
        x = datasets["train"][0][0]
        forces, stress = model.predict_forces(x), model.predict_stress(x)
        assert torch.isfinite(forces).all() and torch.isfinite(stress).all()
        direction = torch.zeros_like(x["positions"])
        direction[0, 1] = 1.0

        def energy(displacement):
            inputs = dict(
                x,
                positions=(x["positions"] + displacement * direction)
                .detach()
                .requires_grad_(True),
            )
            predictions = model(inputs, physical_derivatives=True)
            return model.predictions_to_snapshot(
                predictions, inputs["positions"], inputs["box"], x=inputs
            ).get_energy()

        epsilon = 1e-5
        finite_difference = (energy(epsilon) - energy(-epsilon)) / (2 * epsilon)
        torch.testing.assert_close(
            -(forces * direction).sum(),
            finite_difference.squeeze(),
            rtol=3e-4,
            atol=2e-6,
        )
        (forces.square().sum() + stress.square().sum()).backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
@pytest.mark.parametrize("objective", ["matrix", "irreps", "irrep_parts"])
def test_reference_components_accept_original_block_supervision(
    tmp_path, name, objective
):
    from scripts.train import build_components

    splits = native_data(tmp_path / "data")
    config = OmegaConf.load(document(tmp_path, name, splits))
    config.data.representation = "orbital_blocks"
    config.train_target = "irreps" if objective == "irreps" else "matrix"
    config.train_on_irrep_parts = objective == "irrep_parts"
    config.data.basis = "e3nn"
    config.graph_source = "target_edges"
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, model, datasets, _ = build_components(config)
        model.log_dict = lambda values, **kwargs: None
        loss = model._shared_step(datasets["train"][0], 0, "train")
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    finally:
        torch.set_default_dtype(previous)


def test_original_registered_readout_accepts_packed_shell_supervision(tmp_path):
    from scripts.train import build_components

    splits = native_data(tmp_path / "data")
    config = OmegaConf.load(document(tmp_path, "deeph_e3", splits))
    config.pipeline = None
    config.hidden_irreps = "2x0e+1x1o"
    config.hidden_base_dim = 2
    config.n_radial = 4
    config.radial_layers = [4]
    config.num_layers_gnn = 1
    config.neck_depth = 1
    config.internal_e3mlp_layers = 1
    config.head_e3mlp_layers = 1
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, model, datasets, _ = build_components(config)
        model.log_dict = lambda values, **kwargs: None
        loss = model._shared_step(datasets["train"][0], 0, "train")
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
    finally:
        torch.set_default_dtype(previous)


def test_public_deeph_trainer_matches_norm_clipped_source_updates_and_lr_events(
    tmp_path,
):
    """Exercise real Trainer wiring, not just FeaturePipeline plus a chosen Adam."""
    from scripts.train import build_components
    from data.sample import GraphBatch

    splits = native_data(tmp_path / "data")
    for path in splits["train"]:
        with h5py.File(Path(path) / "hamiltonians.h5", "r+") as handle:
            for key in handle:
                handle[key][...] = np.asarray(handle[key]) + 1000.0
    config_path = document(tmp_path, "deeph", splits)
    config = OmegaConf.load(config_path)
    assert config.training.grad_clip_algorithm == "norm"
    assert list(config.optimization.scheduler.milestones) == [500, 2000, 3000]
    assert list(config.optimization.scheduler.factors) == [0.4, 0.5, 0.5]
    config.training.max_epochs = 3
    # Same controller semantics, with early events to test real Trainer ordering.
    config.optimization.scheduler.milestones = [0, 1]
    config.optimization.scheduler.factors = [0.4, 0.5]
    config.revert_on_spike = False
    config.checkpoint_monitor = "val/loss"
    OmegaConf.save(config, config_path, resolve=False)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _, _, manual, datasets, _ = build_components(config)
        manual.log_dict = lambda *args, **kwargs: None
        optimizer = torch.optim.Adam(
            manual.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8
        )
        batch = GraphBatch([datasets["train"][0], datasets["train"][1]])
        source_events = {0: 0.4, 1: 0.5}
        norms = []
        for epoch in range(3):
            optimizer.zero_grad()
            loss = manual._shared_step(batch, 0, "train")
            loss.backward()
            norms.append(
                float(torch.nn.utils.clip_grad_norm_(manual.parameters(), 4.2))
            )
            optimizer.step()
            if epoch in source_events:
                for group in optimizer.param_groups:
                    group["lr"] *= source_events[epoch]
        assert max(norms) > 4.2  # Clipping is active in this comparison.
        main(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "trained"),
                "--accelerator",
                "cpu",
            ]
        )
        checkpoint = torch.load(
            tmp_path / "trained/latest_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["hyper_parameters"]["cfg"].grad_clip_algorithm == "norm"
        assert checkpoint["optimizer_states"][0]["param_groups"][0][
            "lr"
        ] == pytest.approx(0.0002)
        for name, parameter in manual.named_parameters():
            torch.testing.assert_close(
                checkpoint["state_dict"][name], parameter, atol=1e-9, rtol=1e-9
            )
        for key, state in optimizer.state_dict()["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(
                    checkpoint["optimizer_states"][0]["state"][key][field],
                    value,
                    atol=1e-8,
                    rtol=1e-9,
                )
    finally:
        torch.set_default_dtype(previous)


def test_public_cli_trains_decomposed_tensor_attention_with_distinct_support(tmp_path):
    from net.transforms.composition import ProjectedAttention

    config = OmegaConf.load(
        ROOT / "configs/hybrids/tensor_messages_projected_attention.yaml"
    )
    splits = native_data(tmp_path / "data")
    config.data.splits = splits
    config.data.supports.message.cutoff = 1.15
    config.training.max_epochs = 1
    config.training.dtype = "float64"
    config.training.batch_size = 2
    path = tmp_path / "tensor_attention.yaml"
    OmegaConf.save(config, path, resolve=False)
    previous = torch.get_default_dtype()
    try:
        main(
            [
                "--config",
                str(path),
                "--output",
                str(tmp_path / "run"),
                "--accelerator",
                "cpu",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        checkpoint = torch.load(
            tmp_path / "run/latest_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["epoch_boundary_resume"]["safe_to_resume"]
        assert checkpoint["epoch"] == 1
        for value in checkpoint["state_dict"].values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
        from scripts.train import build_components

        _, _, model, datasets, _ = build_components(config, resume=True)
        model.load_state_dict(checkpoint["state_dict"])
        sample = datasets["train"][0][0]
        assert sample["message_edge_index"].shape[1] < sample["edge_index"].shape[1]
        assert any(
            isinstance(stage, ProjectedAttention)
            for stage in model.pipeline.transforms.values()
        )
        assert list((tmp_path / "logs").rglob("metrics.csv"))
    finally:
        torch.set_default_dtype(previous)
