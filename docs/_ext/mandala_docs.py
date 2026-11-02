"""Directives that render reference material directly from the source code.

``.. config-reference::``
    Every :class:`net.common.Config` field, grouped by the section comments of
    its definition, with defaults and inline notes.

``.. irrep-atlas::``
    Clebsch-Gordan decompositions of orbital-shell pairs and complete atom-pair
    blocks, computed with e3nn and the MANDALA orbital mapper.

``.. transform-registry::``
    All transformation names accepted by the pipeline compiler.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes

ROOT = Path(__file__).resolve().parents[2]


def _render(directive, lines):
    container = nodes.container()
    directive.state.nested_parse(StringList(lines), directive.content_offset, container)
    return container.children


def _table(header, rows, widths=None):
    lines = [".. list-table::", "   :header-rows: 1"]
    if widths:
        lines.append(f"   :widths: {' '.join(map(str, widths))}")
    lines.append("")
    for row in [header, *rows]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])
    lines.append("")
    return lines


class ConfigReference(Directive):
    has_content = False

    def run(self):
        source = (ROOT / "src" / "net" / "common.py").read_text().splitlines()
        tree = ast.parse("\n".join(source))
        config = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Config"
        )
        fields = [node for node in config.body if isinstance(node, ast.AnnAssign)]
        sections, current = {}, "Sections and cutoff"
        cursor = config.lineno
        for node in fields:
            for line in source[cursor : node.lineno - 1]:
                match = re.match(r"\s*#\s*-+\s*(.*?)\s*-+\s*$", line)
                if match and match.group(1):
                    current = match.group(1).strip().rstrip(":").capitalize()
            cursor = node.end_lineno
            name = node.target.id
            default = (
                ast.get_source_segment("\n".join(source), node.value)
                if node.value
                else ""
            )
            default = re.sub(r"\s+", " ", default or "")
            match = re.match(r"field\(default_factory=(?:lambda: )?(.*)\)$", default)
            if match:
                default = match.group(1)
            note = ""
            for line in source[node.lineno - 1 : node.end_lineno]:
                if "#" in line:
                    note = line.split("#", 1)[1].strip()
            annotation = ast.get_source_segment("\n".join(source), node.annotation)
            sections.setdefault(current, []).append(
                (
                    f"``{name}``",
                    f"``{annotation}``",
                    f"``{default}``" if default else "",
                    note,
                )
            )
        lines = []
        for title, rows in sections.items():
            lines += [f".. rubric:: {title}", ""]
            lines += _table(
                ("Field", "Type", "Default", "Note"), rows, (24, 18, 26, 32)
            )
        return _render(self, lines)


class IrrepAtlas(Directive):
    has_content = False
    option_spec = {"shells": directives.unchanged, "species": directives.unchanged}

    def run(self):
        from e3nn import o3

        lines = []
        if "shells" in self.options:
            letters = self.options["shells"].split()
            degree = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
            rows = []
            for a in letters:
                for b in letters:
                    if degree[b] < degree[a]:
                        continue
                    la, lb = degree[a], degree[b]
                    left = o3.Irrep(la, (-1) ** la)
                    right = o3.Irrep(lb, (-1) ** lb)
                    product = o3.Irreps([(1, ir) for ir in left * right])
                    rows.append(
                        (
                            f"{a}-{b}",
                            f"{2 * la + 1} x {2 * lb + 1}",
                            f"``{product}``",
                            str(product.dim),
                        )
                    )
            lines += _table(
                ("Shell pair", "Block", "Irreps", "Dimension"), rows, (14, 14, 52, 14)
            )
        if "species" in self.options:
            import torch
            from core.block_irrep_mapper import BlockIrrepMapper
            from core.orbital_irrep_config import OrbitalIrrepConfig

            spec = dict(item.split("=") for item in self.options["species"].split())
            orbital = OrbitalIrrepConfig.from_dict(
                {element: value.split("+") for element, value in spec.items()}
            )
            mapper = BlockIrrepMapper(orbital, dtype=torch.float64)
            rows = []
            for left in spec:
                for right in spec:
                    key = f"{left}-{right}"
                    rows_, cols = mapper.block_dims(key)
                    irreps = mapper.get_pair_irreps(key)
                    rows.append(
                        (key, f"{rows_} x {cols}", f"``{irreps}``", str(irreps.dim))
                    )
            lines += _table(
                ("Atom pair", "Block", "Irreps", "Dimension"), rows, (14, 14, 52, 14)
            )
        return _render(self, lines)


class TransformRegistry(Directive):
    has_content = False

    def run(self):
        from net.pipeline import _TRANSFORMS

        source = (ROOT / "src" / "net" / "pipeline.py").read_text()
        body = source[
            source.index("def build_transform") : source.index("class _CodecDecode")
        ]
        built_in = set(re.findall(r'name == "([a-z_]+)"', body))
        for group in re.findall(r"name in \{([^}]*)\}", body):
            built_in |= set(re.findall(r'"([a-z_]+)"', group))
        names = sorted(built_in | set(_TRANSFORMS))
        lines = [
            f"The compiler currently accepts **{len(names)}** transformation names:",
            "",
        ]
        lines.append(", ".join(f"``{name}``" for name in names))
        lines.append("")
        return _render(self, lines)


def setup(app):
    app.add_directive("config-reference", ConfigReference)
    app.add_directive("irrep-atlas", IrrepAtlas)
    app.add_directive("transform-registry", TransformRegistry)
    return {"parallel_read_safe": True, "version": "1"}
