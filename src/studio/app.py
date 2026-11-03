"""Flask application serving the Studio API and the built frontend."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory

from studio import analysis, expressions, recipes

WEB = recipes.ROOT / "tools" / "studio" / "web" / "dist"
PLUGINS = recipes.ROOT / "plugins"


def _execute_plugin(path):
    """Import a plugin file afresh, so regenerated blocks re-register."""
    name = f"mandala_studio_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_generated_plugins(directory):
    """Register every generated plugin; returns ``{file: error}`` for failures."""
    failures = {}
    for path in sorted(Path(directory).glob("*.py")):
        try:
            _execute_plugin(path)
        except Exception as error:  # a broken plugin must not stop the editor
            failures[path.name] = str(error)
    return failures


LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _cost_calibrated():
    """The device the cost constants were measured on, or False."""
    from net.cost_model import CONSTANTS_PATH, load_constants

    if not CONSTANTS_PATH.exists():
        return False
    return str(load_constants().get("device") or "measured")


def create_app(*, web=WEB, plugins=PLUGINS):
    app = Flask(__name__, static_folder=None)
    service = analysis.AnalysisService()
    plugin_failures = load_generated_plugins(plugins)

    @app.before_request
    def same_origin_only():
        """Refuse requests a web page on another origin could forge.

        The server executes plugin code, so it accepts only requests addressed
        to a loopback host name, from no origin or a loopback origin, and POST
        bodies declared as JSON (which browsers cannot send cross-origin
        without a preflight this server never grants).
        """
        host = (request.host or "").rsplit(":", 1)[0].strip("[]")
        if host not in LOOPBACK:
            return fail("Studio only answers requests to localhost", status=403)
        origin = request.headers.get("Origin")
        if origin and (urlsplit(origin).hostname or "") not in LOOPBACK:
            return fail("Cross-origin requests are not accepted", status=403)
        if request.method == "POST" and not request.is_json:
            return fail("POST bodies must be application/json", status=415)

    def body():
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    def fail(message, status=400, **extra):
        return jsonify({"error": message, **extra}), status

    @app.get("/api/catalog")
    def catalog():
        shells = recipes.SYSTEMS["Si s2p2d1"]
        return jsonify(
            {
                "transforms": analysis.transform_catalog(),
                "objective_terms": analysis.objective_catalog(),
                "block_library": analysis.block_library_catalog(),
                "inputs": analysis.input_catalog(len(shells)),
                "systems": recipes.SYSTEMS,
                "plugin_failures": plugin_failures,
                "cost_calibrated": _cost_calibrated(),
            }
        )

    @app.get("/api/presets")
    def presets():
        return jsonify(recipes.presets())

    @app.post("/api/import")
    def import_recipe():
        data = body()
        try:
            if data.get("path"):
                document = recipes.load_preset(data["path"])
            else:
                document = recipes.parse_yaml(data.get("yaml", ""))
        except Exception as error:
            return fail(f"Cannot read recipe: {error}")
        try:
            document = recipes.inline_block_libraries(document)
        except ValueError as error:
            return fail(str(error))
        result = service.analyze(
            document, system=data.get("system"), size=data.get("size")
        )
        return jsonify({"document": document, "analysis": result})

    @app.post("/api/analyze")
    def analyze():
        data = body()
        document = data.get("document")
        if not isinstance(document, dict):
            return fail("document must be a mapping")
        return jsonify(
            service.analyze(document, system=data.get("system"), size=data.get("size"))
        )

    @app.post("/api/export")
    def export():
        data = body()
        document = data.get("document")
        if not isinstance(document, dict):
            return fail("document must be a mapping")
        return jsonify({"yaml": recipes.dump_yaml(document, header=data.get("header"))})

    @app.post("/api/expression/check")
    def check_expression():
        data = body()
        return jsonify(
            expressions.check(
                data.get("name", ""),
                data.get("inputs", []),
                data.get("source", ""),
                data.get("outputs", ["out"]),
                data.get("summary", ""),
                data.get("category", "custom"),
            )
        )

    @app.post("/api/plugin")
    def plugin():
        data = body()
        result = expressions.check(
            data.get("name", ""),
            data.get("inputs", []),
            data.get("source", ""),
            data.get("outputs", ["out"]),
            data.get("summary", ""),
            data.get("category", "custom"),
        )
        if not result["ok"]:
            return fail(result["error"]["message"], error_detail=result["error"])
        from net.inspection import clear_cache
        from net.registry import registered_transforms

        plugins.mkdir(parents=True, exist_ok=True)
        path = plugins / f"{data['name']}.py"
        if data["name"] in registered_transforms() and not path.exists():
            return fail(
                f"{data['name']!r} is a framework transformation; choose another name"
            )
        handle, temporary = tempfile.mkstemp(dir=plugins, suffix=".tmp")
        with os.fdopen(handle, "w") as stream:
            stream.write(result["code"])
        os.replace(temporary, path)
        try:
            _execute_plugin(path)
        except Exception as error:
            path.unlink(missing_ok=True)
            return fail(f"Generated plugin failed to load: {error}", status=500)
        clear_cache()
        return jsonify(
            {
                "path": recipes.relative_path(path),
                "code": result["code"],
                "transforms": analysis.transform_catalog(),
            }
        )

    @app.get("/")
    @app.get("/<path:asset>")
    def frontend(asset="index.html"):
        if not web.exists():
            return (
                "The Studio frontend is not built. Run `npm install && npm run build` "
                "in tools/studio/web.",
                503,
            )
        if not (web / asset).is_file():
            asset = "index.html"
        return send_from_directory(web, asset)

    return app
