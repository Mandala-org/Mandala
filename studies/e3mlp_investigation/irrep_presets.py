from __future__ import annotations

from e3nn.o3 import Irreps


PRELIMINARY_HIDDEN_IRREPS = Irreps("16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e")
SILICON_HIDDEN_IRREPS = Irreps("32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e")
SILICON_DOUBLED_HIDDEN_IRREPS = Irreps(
    "64x0e+64x0o+32x1e+32x1o+16x2e+16x2o+16x3e+16x3o+16x4e"
)


def get_hidden_irreps(preset: str) -> Irreps:
    preset = preset.lower()
    if preset == "preliminary":
        return PRELIMINARY_HIDDEN_IRREPS
    if preset == "silicon":
        return SILICON_HIDDEN_IRREPS
    if preset == "silicon_doubled":
        return SILICON_DOUBLED_HIDDEN_IRREPS
    raise ValueError(f"Unknown irreps preset '{preset}'")
