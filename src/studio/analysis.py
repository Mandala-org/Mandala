"""Recipe analysis for the editor: stages, contracts, costs and the objective.

Analyses run one at a time (the stage cache and torch state are shared). When
several requests of one editor page queue up, only the newest is computed;
older ones return ``{"stale": true}`` so an editor sending one request per edit
stays responsive. Imports are always answered.
"""

from __future__ import annotations

import threading
import time
import traceback

from studio.recipes import check_plugin_paths, framework_document, system_for


def objective_catalog():
    try:
        from net.objectives import objective_catalog as catalog
    except Exception:  # objectives are optional for the editor
        return []
    try:
        return catalog()
    except Exception:
        return []


def transform_catalog():
    from net.registry import transform_catalog as catalog

    return catalog()


def block_library_catalog():
    from net.inspection import block_library_catalog as catalog

    return catalog()


def input_catalog(num_species):
    from net.features import default_input_specs

    return {
        slot: spec.to_json() for slot, spec in default_input_specs(num_species).items()
    }


def analyze_objective(document, cfg):
    """Explicit or flat-lowered objective, with validation messages."""
    result = {"objective": None, "lowered": False, "errors": [], "labels": []}
    try:
        from net import objectives
    except Exception:
        result["errors"].append(
            "Objective terms are not available in this framework version"
        )
        result["objective"] = document.get("objective")
        return result
    specification = document.get("objective")
    explicit = isinstance(specification, dict) and "terms" in specification
    if not explicit and cfg is not None and hasattr(objectives, "default_objective"):
        try:
            specification = objectives.default_objective(cfg)
            result["lowered"] = True
        except Exception as error:
            result["errors"].append(f"Default objective: {error}")
            return result
    result["objective"] = specification
    if not isinstance(specification, dict) or "terms" not in specification:
        return result
    try:
        if specification.get("expression"):
            _, labels = objectives.parse_expression(specification["expression"])
            result["labels"] = sorted(labels)
        objectives.check_objective(specification)
    except Exception as error:
        result["errors"].append(str(error))
    return result


class AnalysisService:
    """Serialized analysis of recipe documents, newest-wins per editor.

    Each editor page (``client``) sees only its own requests superseded, and a
    request that must be answered (an import) is never answered as stale.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counter = threading.Lock()
        self._generations = {}

    def analyze(self, document, *, system=None, size=None, client=None, may_skip=True):
        key = str(client or "")
        with self._counter:
            mine = self._generations.get(key, 0) + 1
            self._generations[key] = mine
        with self._lock:
            if may_skip and mine != self._generations.get(key):
                return {"stale": True}
            return self._analyze(document, system, size)

    def _analyze(self, document, system, size):
        from net.cost_model import SystemSize
        from net.inspection import analyze_recipe, resolve_recipe

        started = time.perf_counter()
        document = framework_document(document)
        try:
            check_plugin_paths(document)
            shells = system_for(document, system)
            size = SystemSize(**(size or {}))
        except (ValueError, TypeError, AttributeError) as error:
            return {
                "stages": [],
                "slots": {},
                "inputs": {},
                "groups": {},
                "outputs": {},
                "errors": [{"stage": "request", "message": str(error)}],
                "totals": {"parameters": 0, "seconds": 0.0},
                "pipeline": None,
            }
        try:
            analysis = analyze_recipe(document, species_shells=shells, size=size)
        except Exception as error:
            analysis = {
                "stages": [],
                "slots": {},
                "inputs": {},
                "groups": {},
                "outputs": {},
                "errors": [{"stage": "analysis", "message": f"{error}"}],
                "totals": {"parameters": 0, "seconds": 0.0},
                "pipeline": None,
                "trace": traceback.format_exc(),
            }
        cfg = None
        if analysis.get("pipeline") is not None:
            try:
                cfg, _ = resolve_recipe(document, shells)
            except Exception:
                cfg = None
        analysis["objective"] = analyze_objective(document, cfg)
        analysis["system"] = {"species_shells": shells, **size.__dict__}
        analysis["flat"] = not bool(document.get("pipeline"))
        analysis["elapsed"] = time.perf_counter() - started
        return analysis
