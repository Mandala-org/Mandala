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

    # --------------------------------------------------------------------- API
    @classmethod
    def from_dict(cls, dct: Dict[str, Sequence[str] | str]) -> "OrbitalIrrepConfig":
        """
        Build an :class:`OrbitalIrrepConfig` from a *python* mapping
        ``{element: spec}``.

        Accepted **spec** formats
        ------------------------
        1. **List / tuple** of tokens::

               {"Si": ["2x0e", "2x1o", "1x2e"]}

        2. **Full Irreps string** (``+``‑separated; whitespace ignored)::

               {"Si": "2x0e + 2x1o + 1x2e"}

        3. **Compact orbital string** – concatenation of ``<n><orbital>`` where
           *orbital* is one of ``s p d f g h i k l m`` (case‑insensitive).
           Example::

               {"Si": "3s2p2d1f"}     # → 3x0e + 2x1o + 2x2e + 1x3o
        """
        if not isinstance(dct, dict):
            raise OrbitalIrrepConfigError("Input must be a dict[element -> irreps]")

        # map orbital letter → ℓ
        _orbital_to_l = {
            "s": 0,
            "p": 1,
            "d": 2,
            "f": 3,
            "g": 4,
            "h": 5,
            "i": 6,
            "k": 7,
            "l": 8,
            "m": 9,
        }

        element_to_irreps: Dict[str, Irreps] = {}
        for element, spec in dct.items():
            # ---------- validate key --------------------------------------
            if not isinstance(element, str):
                raise OrbitalIrrepConfigError(
                    f"Element keys must be str (got {type(element)})"
                )

            # ---------- canonicalise spec to '+' notation -----------------
            if isinstance(spec, str):
                expr = spec.replace(" ", "")  # drop whitespace

                # detect compact syntax: only digits / orbital letters, no 'x'/'e'/'o'/'+'.
                if all(ch.isdigit() or ch.lower() in _orbital_to_l for ch in expr):
                    # parse e.g. "3s2p2d1f"
                    import re

                    tokens = []
                    for num, orb in re.findall(
                        r"(\d*)([spdfghiklm])", expr, flags=re.I
                    ):
                        mul = int(num) if num else 1
                        l = _orbital_to_l[orb.lower()]
                        parity = "e" if l % 2 == 0 else "o"
                        tokens.append(f"{mul}x{l}{parity}")
                    expr = "+".join(tokens)
                # else: assume user already wrote an Irreps expression
            elif isinstance(spec, (list, tuple)):
                expr = "+".join(str(s).replace(" ", "") for s in spec)
            else:
                raise OrbitalIrrepConfigError(
                    f"Value for element '{element}' must be str or list/tuple, "
                    f"got {type(spec)}"
                )

            # ---------- parse with e3nn -----------------------------------
            try:
                irreps = Irreps(expr)
            except Exception as exc:  # pragma: no cover
                raise OrbitalIrrepConfigError(
                    f"Failed to parse irreps for element '{element}': {exc}"
                ) from exc

            # ---------- sanity: ℓ limit -----------------------------------
            l_max_seen = max(ir.l for _, ir in irreps)
            if l_max_seen > 10:
                raise OrbitalIrrepConfigError(
                    f"Element '{element}': l={l_max_seen} orbitals not supported (max 10)"
                )

            element_to_irreps[element] = irreps

        return cls(element_to_irreps)

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, List[str]]:
        """
        Return a plain-python mapping of the *original* irrep strings so the
        config can be safely written to YAML / JSON.
        """
        return {
            el: [f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}" for mul, ir in irreps]
            for el, irreps in self.element_to_irreps.items()
        }

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
