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

from scripts.train_modular import main

ROOT = Path(__file__).resolve().parents[2]


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
            specification = json.loads(
                (tmp_path / "run/specification.json").read_text()
            )
            affine = specification["statistics"]["output_affine"]
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
    from scripts.train_modular import dataset_paths

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
    from scripts.train_modular import dataset_paths

    with pytest.raises(ValueError, match=match):
        dataset_paths({"data": {"root": "/path/never/loaded"}, "training": options})


@pytest.mark.parametrize("name", ["deeph", "deeph_e3", "mace_h"])
def test_complex_soc_presets_native_dataset_through_lightning_fit(tmp_path, name):
    from scripts.train_modular import build_components
    from data.operator_dataset import collate_operator_samples

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
        batch = collate_operator_samples(datasets["train"].samples)
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
        from scripts.predict_modular import main as predict_main

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
    from scripts.train_modular import dataset_paths

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
    with pytest.raises(ValueError, match="No native operator snapshots"):
        dataset_paths(document)


@pytest.mark.parametrize(
    "radius", [None, 0, -1.0, float("nan"), float("inf"), True, "2"]
)
def test_radius_training_configuration_requires_finite_positive_radius(radius):
    from scripts.train_modular import dataset_paths

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
