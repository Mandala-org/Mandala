"""Run MANDALA Studio: ``python -m studio [--port 8765] [--host 127.0.0.1]``."""

import argparse
import os

from studio.app import create_app


def main():
    parser = argparse.ArgumentParser(description="MANDALA Studio: visual recipe editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # Relative plugin paths in recipes are resolved against the repository root.
    from studio.recipes import ROOT

    os.chdir(ROOT)
    create_app().run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
