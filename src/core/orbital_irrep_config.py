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

# ! Add: block_dims

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import yaml
from e3nn.o3 import Irreps

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.openmx_info_parser import InfoOutData  # type: ignore[import-untyped]


class OrbitalIrrepConfigError(ValueError):
    """Raised when the YAML / dict does not meet the expected schema."""


@dataclass(slots=True)
class OrbitalIrrepConfig:
    """Holds per-element orbital irreps and exposes convenience accessors."""

    element_to_irreps: Dict[str, Irreps] = field(default_factory=dict)
    _elem_dim_cache: Dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

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

        2. **Full Irreps string** (``+``-separated; whitespace ignored)::

               {"Si": "2x0e + 2x1o + 1x2e"}

        3. **Compact orbital string** - concatenation of ``<n><orbital>`` where
           *orbital* is one of ``s p d f g h i k l m`` (case-insensitive).
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

    # ──────────────────────────────────────────────────────────────────
    # NEW STATIC HELPERS
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def merge(configs: Sequence["OrbitalIrrepConfig"]) -> "OrbitalIrrepConfig":
        """
        Union-merge several configs **without duplicating BlockIrrepMappers**.

        * If an element appears in more than one config its irreps must be
          *identical* – otherwise we raise to avoid silent mismatches.
        """
        if len(configs) == 0:
            raise ValueError("merge() needs at least one OrbitalIrrepConfig")

        merged: Dict[str, Sequence[str] | str] = {}
        for cfg in configs:
            for el, spec in cfg.to_dict().items():
                if el in merged and merged[el] != spec:
                    raise OrbitalIrrepConfigError(
                        f"Conflicting irreps for element '{el}': "
                        f"{merged[el]}  vs  {spec}"
                    )
                merged[el] = spec
        return OrbitalIrrepConfig.from_dict(merged)

    # -----------------------------------------------------------------
    @staticmethod
    def from_info_list(info_list: Sequence["InfoOutData"]) -> "OrbitalIrrepConfig":
        """
        Convenience helper – derive a **project-wide** config directly from a
        collection of :class:`data.openmx_info_parser.InfoOutData` objects.
        """
        collected: Dict[str, Sequence[str] | str] = {}
        for info in info_list:
            for el, spec in info.orbital_set.items():
                if el in collected and collected[el] != spec:
                    raise OrbitalIrrepConfigError(
                        f"Conflicting orbital spec for element '{el}': "
                        f"{collected[el]}  vs  {spec}"
                    )
                collected[el] = spec
        return OrbitalIrrepConfig.from_dict(collected)

    # ------------------------------------------------------------------ dim helper
    def block_dims(self, pair: Tuple[str, str] | str) -> Tuple[int, int]:
        """
        Return ``(d_i, d_j)`` – the orbital dimensions of the two atoms
        that form *one* matrix block.

        Parameters
        ----------
        pair
            Either a hyphen-joined string ``"Si-H"`` or a two-tuple
            ``("Si", "H")``.
        """
        # canonicalise ----------------------------------------------
        if isinstance(pair, str):
            if "-" not in pair:
                raise ValueError("String key must look like 'A-B'")
            el_i, el_j = pair.split("-", 1)
        elif isinstance(pair, tuple) and len(pair) == 2:
            el_i, el_j = pair
        else:
            raise ValueError("pair must be tuple(str,str) or 'A-B' string")

        # cache per element for efficiency --------------------------
        if not self._elem_dim_cache:
            self._elem_dim_cache = {
                el: irr.dim for el, irr in self.element_to_irreps.items()
            }

        try:
            return self._elem_dim_cache[el_i], self._elem_dim_cache[el_j]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unknown element in pair: {pair}") from exc
