"""Installs requirements.txt into imitationagent_training/python_embedded (pip --target) for the Unity retrain.

Run it manually once, and again whenever requirements.txt changes, with the interpreter the training should use
(>= 3.11 for zarr 3; compiled packages only load in the Python minor version they were installed for):

    /opt/homebrew/bin/python3 imitationagent_training/training_pipeline/bootstrap_env.py [--force]

It writes python_embedded/.bootstrap.json (interpreter path, version, sha256 of requirements.txt). Unity's
RetrainPipeline runs that interpreter with PYTHONPATH=python_embedded and refuses to start when the marker is missing
or its hash no longer matches requirements.txt. Unity itself never installs packages.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.abspath(os.path.join(HERE, "..", "python_embedded"))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
MARKER = os.path.join(TARGET_DIR, ".bootstrap.json")
MIN_PYTHON = (3, 11)


def requirements_sha256() -> str:
    with open(REQUIREMENTS, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_marker():
    try:
        with open(MARKER) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Sync python_embedded with requirements.txt")
    parser.add_argument("--force", action="store_true", help="Reinstall even if python_embedded is up to date.")
    args = parser.parse_args()

    if sys.version_info < MIN_PYTHON:
        sys.exit(f"[BOOTSTRAP] Python {sys.version.split()[0]} is too old; run this with Python >= "
                 f"{'.'.join(map(str, MIN_PYTHON))} (e.g. /opt/homebrew/bin/python3).")

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    wanted = requirements_sha256()
    marker = read_marker()
    if (not args.force and marker and marker.get("requirements_sha256") == wanted
            and marker.get("python_version") == version and os.path.exists(marker.get("python", ""))):
        print(f"[BOOTSTRAP] {TARGET_DIR} is up to date (Python {version}, {marker['python']}).")
        return

    # A clean install: pip --target does not remove packages that were dropped or replaced in requirements.txt.
    if os.path.isdir(TARGET_DIR):
        print(f"[BOOTSTRAP] Removing old {TARGET_DIR} ...")
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)
    print(f"[BOOTSTRAP] Installing {REQUIREMENTS} into {TARGET_DIR} with {sys.executable} (Python {version}) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", TARGET_DIR, "-r", REQUIREMENTS])

    with open(MARKER, "w") as f:
        json.dump({"python": sys.executable, "python_version": version, "requirements_sha256": wanted}, f, indent=2)
    print(f"[BOOTSTRAP] Done. Unity will train with {sys.executable} + PYTHONPATH={TARGET_DIR}.")


if __name__ == "__main__":
    main()
