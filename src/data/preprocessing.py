"""Configurable Snapshot preprocessing shared by DatasetFactory clients.

The input file format and model name do not select a training implementation.
This processor chooses a declared operator representation and graph policies.
"""

from pathlib import Path
import torch

from data.sample import GraphSample
from utils.units import HARTREE_TO_EV


class PackedOperatorPreprocessing:
    def __init__(self, configuration):
        self.document = configuration
        self.configuration = None
        self.codec = None

    def signature(self):
        return self.configuration or self.document

    def prepare(self, mapper, training_metadata, *, configuration, layout):
        self.configuration, self.codec = configuration, layout

    def __call__(
        self,
        snapshot,
        *,
        matrix_path,
        info_path,
        snapshot_label="snapshot",
        split="train",
    ):
        from data.operator_features import build_operator_sample

        if snapshot.hamiltonian.basis != "openmx":
            snapshot = snapshot.to_openmx()
        data = self.configuration["data"]
        dtype = torch.get_default_dtype()
        atoms = snapshot.hamiltonian.atoms
        shells = {
            species: tuple(ir.l for mul, ir in irreps for _ in range(mul))
            for species, irreps in snapshot.hamiltonian.orbital_cfg.element_to_irreps.items()
        }
        frame_meta = {
            "atoms": atoms,
            "species_shells": shells,
            "positions": snapshot.positions.to(torch.float64),
            "box": snapshot.box.to(torch.float64),
            "spinful": bool(getattr(snapshot.info, "is_spinful", False)),
        }
        meta = dict(
            frame_meta,
            positions=snapshot.positions.to(dtype),
            box=snapshot.box.to(dtype),
        )
        available = set(
            getattr(
                snapshot.info,
                "available_matrices",
                {"hamiltonian", "density", "overlap"},
            )
        )

        def entries(name):
            matrix = snapshot[name]
            result = {}
            for key, (pair, index) in matrix.lookup.items():
                block = matrix.pair_blocks[pair][index]
                tensor_dtype = (
                    (torch.complex128 if dtype == torch.float64 else torch.complex64)
                    if block.is_complex()
                    else dtype
                )
                value = (
                    block.to(torch.complex128 if block.is_complex() else torch.float64)
                    * HARTREE_TO_EV
                    if name == "hamiltonian"
                    else block
                ).to(tensor_dtype)
                # Snapshot's physical operators use Hartree; this target policy
                # requests eV without rewriting the source files.
                result[key] = value
            return dict(
                sorted(
                    result.items(),
                    key=lambda item: str(
                        [*item[0][:3], item[0][3] + 1, item[0][4] + 1]
                    ),
                )
            )

        names = data.get("matrix_files", {"hamiltonian": "hamiltonians.h5"})
        missing = set(names) - available
        if missing:
            raise ValueError(f"Requested operators are absent: {sorted(missing)}")
        matrices = {name: entries(name) for name in names}
        support_name = (
            data.get("support_operator")
            or {filename: operator for operator, filename in names.items()}.get(
                data.get("support_file", names.get("hamiltonian", "hamiltonians.h5"))
            )
            or {
                "hamiltonians.h5": "hamiltonian",
                "overlaps.h5": "overlap",
                "density_matrixs.h5": "density",
            }.get(data.get("support_file", "hamiltonians.h5"))
        )
        support = {}
        if data.get("graph_source", "stored") == "stored":
            if support_name is not None:
                if support_name not in available:
                    raise ValueError(
                        f"Stored graph support operator is absent: {support_name}"
                    )
                support = entries(support_name)
            else:
                from data.deeph_e3_parser import read_periodic_blocks

                support = read_periodic_blocks(
                    Path(info_path).parent / data["support_file"],
                    num_atoms=len(atoms),
                    dtype=dtype,
                )
        stored_frames = None
        if data.get("frame_source", "derive") == "stored":
            from data.deeph_e3_parser import read_periodic_blocks

            stored_frames = read_periodic_blocks(
                Path(info_path).parent / "rc.h5", num_atoms=len(atoms), dtype=dtype
            )
        record = build_operator_sample(
            meta,
            matrices,
            support,
            self.codec,
            species=list(self.configuration["species"]["order"]),
            split=split,
            sample_id=str(Path(matrix_path).parent.resolve()),
            local_coordinates=data.get("local_coordinates", False),
            num_l=data.get("num_l", 5),
            target_frame=data.get("target_frame", "global"),
            graph_source=data.get("graph_source", "stored"),
            graph_radius=data.get("graph_radius"),
            dtype=dtype,
            frame_meta=frame_meta,
            stored_frames=stored_frames,
            supports=data.get("supports"),
            neighborhood_support=data.get("neighborhood_support", "output"),
        )
        return GraphSample(record.pop("inputs"), record.pop("targets"), record)
