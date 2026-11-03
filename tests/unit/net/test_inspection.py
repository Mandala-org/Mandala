"""Recipe analysis reports stages, provenance, sizes, costs and errors."""

import json

from net.inspection import analyze_pipeline, analyze_recipe

SHELLS = {"H": [0, 1], "O": [0, 1, 2]}
FLAT = {
    "cutoff_radius": 3.0,
    "l_max": 2,
    "hidden_irreps": "8x0e+4x1o+2x2e",
    "hidden_base_dim": 8,
    "n_radial": 6,
    "radial_layers": [8],
    "num_layers_gnn": 1,
    "matrix_targets": ["hamiltonian"],
}


def test_flat_configuration_analysis_matches_the_built_model():
    from net.common import Config
    from net.e3gnn import E3GNN
    from net.inspection import orbital_mapper

    result = analyze_recipe(FLAT, species_shells=SHELLS)
    json.dumps(result)
    assert not result["errors"]
    model = E3GNN(orbital_mapper(SHELLS), Config(**{**FLAT, "verbosity": 0}))
    assert result["totals"]["parameters"] == sum(p.numel() for p in model.parameters())
    stages = {stage["id"]: stage for stage in result["stages"]}
    assert stages["node_update_0/pre"]["inferred"] == ["irreps_hidden", "irreps_out"]
    assert stages["radial"]["static"] and stages["radial"]["parameters"] == 0
    assert all(stage["seconds"] > 0 for stage in result["stages"])
    assert result["groups"]["node_update_0"]["block"] == "node_update"
    assert result["outputs"] == {"hamiltonian": "hamiltonian_vectors"}
    assert result["inputs"]["edge_lengths"]["used"]


def test_errors_are_collected_per_stage_instead_of_raised():
    pipeline = {
        "stages": [
            {
                "id": "lin",
                "transform": "irrep_linear",
                "inputs": ["edge_lengths"],
                "outputs": ["x"],
                "options": {"irreps_out": "2x1o"},
            },
            {
                "id": "sh",
                "transform": "spherical_harmonics",
                "inputs": ["node_one_hot"],
                "outputs": ["y"],
                "options": {"irreps": "0e+1o"},
            },
        ],
        "outputs": {"hamiltonian": "missing"},
    }
    result = analyze_pipeline(pipeline, species_shells=SHELLS)
    messages = {error["stage"]: error["message"] for error in result["errors"]}
    assert set(messages) == {"lin", "sh", "outputs"}
    assert "global irrep" in messages["lin"] and "Cartesian" in messages["sh"]


def test_unresolvable_configuration_is_reported():
    result = analyze_recipe({"no_such_field": 1}, species_shells=SHELLS)
    assert result["errors"][0]["stage"] == "configuration"
