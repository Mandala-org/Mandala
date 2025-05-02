"""
irreps_builder.py
=================

Automatic helper that **derives all intermediate Irreps strings** from a *very
small* set of user hyper-parameters.

Purpose
-------
* Centralise how hidden widths are computed (so the same rule is used by the
  model, the head registry, and unit-tests).
* Provide quick **sanity checks** that required Wigner paths exist.

Rules implemented here
----------------------
* Hidden scalar/vector width *decays by factor 2* with every increase in `ell`.
  Example (`base_dim=64, l_max=3`) ::

      l = 0  →  64x0e  + 64x0o
      l = 1  →  32x1e  + 32x1o
      l = 2  →  16x2e  + 16x2o
      l = 3  →   8x3e  +  8x3o

* Both **even** and **odd** parity channels are included for every `ell`
  because a real-orbital Hamiltonian couples even↔odd via translations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from e3nn.o3 import Irrep, Irreps


class IrrepsBuilderError(RuntimeError):
    """Raised when an illegal or impossible irrep request is detected."""


# --------------------------------------------------------------------------- #
def _make_even_odd_mul_str(mul: int, ell: int) -> str:
    """Return f\"{mul}x{ell}e+{mul}x{ell}o\" (handles mul==0)."""
    if mul <= 0:
        return ""
    return f"{mul}x{ell}e+{mul}x{ell}o"


@dataclass(slots=True)
class IrrepsAutoBuilder:
    """
    Convenience builder for hidden features and for Wigner-path sanity checks.

    Parameters
    ----------
    l_max
        Highest ell carried by *learned* features (excludes SH basis of geometry).
    base_dim
        Multiplicity at ell=0.  Multiplicity for ell>0 is `base_dim // (2**ell)`.
    """

    l_max: int
    base_dim: int

    # declare the cache as a slot
    _hidden_cache: Irreps | None = field(init=False, repr=False, default=None)

    # ------------------ derived properties --------------------------------- #
    @property
    def hidden_irreps(self) -> Irreps:
        """Computed once, cached on the instance."""
        if self._hidden_cache is None:
            parts: List[str] = []
            for ell in range(self.l_max + 1):
                mul = max(self.base_dim // (2**ell), 1)
                parts.append(_make_even_odd_mul_str(mul, ell))
            ir_string = "+".join(filter(None, parts))
            # now that _hidden_cache is a real slot, direct assignment works
            self._hidden_cache = Irreps(ir_string).simplify()
        return self._hidden_cache  # type: ignore[attr-defined]

    @property
    def sh_irreps(self) -> Irreps:
        """
        Spherical-harmonics irreps used for geometric embedding
        (`e3nn.o3.spherical_harmonics`).
        """
        return Irreps.spherical_harmonics(self.l_max)

    # ------------------ tensor-product sanity ------------------------------ #
    @staticmethod
    def tp_path_exists(irreps_in1: Irreps, irreps_in2: Irreps, ir_out: Irrep) -> bool:
        """Return *True* iff ir_out appears in the Clebsch-Gordan product."""
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        ir_out = Irrep(ir_out)

        for _, ir1 in irreps_in1:
            for _, ir2 in irreps_in2:
                if ir_out in ir1 * ir2:
                    return True
        return False

    # ------------------ validation helpers --------------------------------- #
    def require_tp_path(
        self, irreps_in1: Irreps, irreps_in2: Irreps, desired: Irreps
    ) -> None:
        """
        Raise :class:`IrrepsBuilderError` if **any** component in *desired*
        cannot be produced by `irreps_in1 ⊗ irreps_in2`.
        """
        for _, ir in Irreps(desired):
            if not self.tp_path_exists(irreps_in1, irreps_in2, ir):
                raise IrrepsBuilderError(
                    f"Tensor product {irreps_in1} ⊗ {irreps_in2} cannot produce {ir}"
                )

    # ------------------ short helpers used elsewhere ----------------------- #
    def even_scalar_mul(self) -> int:
        """Multiplicity of 0e scalars in the hidden representation."""
        for mul, ir in self.hidden_irreps:
            if ir.l == 0 and ir.p == 1:
                return mul
        raise RuntimeError(
            "builder produced hidden irreps without 0e scalars"
        )  # pragma: no cover

    def summary(self) -> str:  # pragma: no cover
        return (
            f"Auto-builder(l_max={self.l_max}, base_dim={self.base_dim}) ->\n"
            f"    hidden   : {self.hidden_irreps}\n"
            f"    SH basis : {self.sh_irreps}"
        )
