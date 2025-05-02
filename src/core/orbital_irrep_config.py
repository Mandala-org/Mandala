"""
orbital_irrep_config.py
-----------------------

Small, self-contained helper for parsing + validating the *user section*
that specifies **which irreducible representations belong to the atomic
orbitals of every chemical element**.

Example accepted YAML fragment
──────────────────────────────
orbitals:
  Si: ["2x0e", "2x1o", "1x2e"]   # two s, two p, one d
  H:  ["1x0e"]                   # one s
  O:
    - 2x0e
    - 2x1o
    - 1x2e
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import yaml
from e3nn.o3 import Irreps


class OrbitalIrrepConfigError(ValueError):
    """Raised when the YAML / dict does not meet the expected schema."""


@dataclass(slots=True)
class OrbitalIrrepConfig:
    """Holds per-element orbital irreps and exposes convenience accessors."""

    element_to_irreps: Dict[str, Irreps] = field(default_factory=dict)

    # --------------------------------------------------------------------- API
    @classmethod
    def from_yaml(cls, yaml_str: str | bytes) -> "OrbitalIrrepConfig":
        """Parse YAML string or bytes into a validated instance."""
        try:
            payload = yaml.safe_load(yaml_str)
        except yaml.YAMLError as exc:  # pragma: no cover
            raise OrbitalIrrepConfigError(f"Invalid YAML: {exc}") from exc

        if not isinstance(payload, dict) or "orbitals" not in payload:
            raise OrbitalIrrepConfigError(
                "YAML must contain a top-level 'orbitals' mapping"
            )

        return cls.from_dict(payload["orbitals"])

    @classmethod
    def from_dict(cls, dct: Dict[str, Sequence[str]]) -> "OrbitalIrrepConfig":
        """Validate and build from python dict {element: list_of_irrep_strings}."""
        if not isinstance(dct, dict):
            raise OrbitalIrrepConfigError("Input must be a dict[str, list[str]]")

        element_to_irreps: Dict[str, Irreps] = {}
        for element, irrep_list in dct.items():
            if not isinstance(element, str):
                raise OrbitalIrrepConfigError(
                    f"Element keys must be str ('{element}' is not)"
                )

            if not isinstance(irrep_list, (list, tuple)):
                raise OrbitalIrrepConfigError(
                    f"Value for element '{element}' must be a list/tuple, got {type(irrep_list)}"
                )

            try:
                # Join list into '+' string e.g. ["1x0e", "1x1o"] -> "1x0e+1x1o"
                irreps = Irreps("+".join(str(s) for s in irrep_list))
            except Exception as exc:
                raise OrbitalIrrepConfigError(
                    f"Failed to parse irreps for element '{element}': {exc}"
                ) from exc

            # quick sanity: only accept orbitals up to l=4 (g) by default
            l_max_seen = max(ir.l for _, ir in irreps)
            if l_max_seen > 10:
                raise OrbitalIrrepConfigError(
                    f"Element '{element}': l={l_max_seen} orbitals not supported (max 10)"
                )

            element_to_irreps[element] = irreps

        return cls(element_to_irreps)

    # ---------------------------------------------------------------- helpers
    def max_l(self) -> int:
        """Largest angular momentum encountered across all elements."""
        return max(
            ir.l for irreps in self.element_to_irreps.values() for _, ir in irreps
        )

    def elements(self) -> List[str]:
        """Return the list of element symbols in deterministic order."""
        return sorted(self.element_to_irreps)

    # ---------------------------------------------------------------- dunder
    def __repr__(self) -> str:  # pragma: no cover
        tbl = "\n".join(f"  {el}: {irr}" for el, irr in self.element_to_irreps.items())
        return f"OrbitalIrrepConfig(\n{tbl}\n)"
