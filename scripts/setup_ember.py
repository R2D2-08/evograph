#!/usr/bin/env python3
"""Provision or verify the dedicated EMBER environment using only the stdlib."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11.14"


def run(command):
    print("+ " + shlex.join(command), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv", type=Path, default=PROJECT_ROOT / ".venv-ember",
        help="Dedicated environment path (relative paths are relative to the project root)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Verify an existing environment without downloads or package changes",
    )
    args = parser.parse_args()
    venv = args.venv.expanduser()
    if not venv.is_absolute():
        venv = PROJECT_ROOT / venv
    venv = Path(os.path.abspath(venv))
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment_files = PROJECT_ROOT / "environments/ember"

    try:
        uv = shutil.which("uv")
        if not args.verify_only and uv is None:
            raise RuntimeError(
                "uv is not on PATH. Install uv using https://docs.astral.sh/uv/getting-started/installation/ "
                "and rerun this command (tested with uv 0.12.21)."
            )
        if venv.exists() or venv.is_symlink():
            if not (venv / "pyvenv.cfg").is_file() or not python.is_file():
                raise RuntimeError(
                    f"{venv} is not an intact virtual environment. It was left unchanged; "
                    "choose a new dedicated path with --venv."
                )
        elif args.verify_only:
            raise RuntimeError(
                f"{venv} does not exist. Run this script without --verify-only first."
            )
        else:
            run([uv, "venv", "--no-project", "--python", PYTHON_VERSION, str(venv)])

        inspection = subprocess.run(
            [str(python), "-c", "import json, sys; print(json.dumps({"
             "'version': '.'.join(map(str, sys.version_info[:3])), "
             "'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        info = json.loads(inspection.stdout)
        if (info["prefix"] == info["base_prefix"]
                or Path(info["prefix"]).resolve() != venv.resolve()):
            raise RuntimeError(
                f"{python} does not run inside {venv}. No packages were changed; "
                "choose a new dedicated path with --venv."
            )
        if info["version"] != PYTHON_VERSION:
            raise RuntimeError(
                f"{venv} uses Python {info['version']}; exactly {PYTHON_VERSION} is required. "
                "The existing environment was left unchanged. Choose a new path with --venv; "
                "this script never deletes or recreates an existing environment."
            )
        if not args.verify_only:
            run([
                uv, "pip", "sync", "--python", str(python), "--require-hashes",
                "--build-constraints", str(environment_files / "build-constraints.txt"),
                "--strict", str(environment_files / "requirements.lock"),
            ])
        run([str(python), str(PROJECT_ROOT / "scripts/verify_ember.py")])
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"EMBER setup failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.rstrip(), file=sys.stderr)
        print("See environments/ember/README.md for prerequisites and troubleshooting.", file=sys.stderr)
        return 1

    print("\nRead-only extraction with the dedicated interpreter:")
    print(shlex.join([
        str(python), str(PROJECT_ROOT / "run_multimodal.py"), "extract-static",
        "--binary", "/path/to/sample.exe", "--out", "/path/to/static.json",
    ]))
    print("\nOptional bridge from the separate, modern model environment:")
    print("export EVOGRAPH_EMBER_PYTHON=" + shlex.quote(str(python)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
