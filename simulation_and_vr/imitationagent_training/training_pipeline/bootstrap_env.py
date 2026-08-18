import os
import re
import subprocess
import sys


def pkg_import_name(requirement: str) -> str:
    """Convert a pip requirement such as 'torch>=2.8.0,<3.0.0' to a safe import name."""
    requirement = requirement.strip()
    match = re.match(r"^[A-Za-z0-9_.-]+", requirement)
    if not match:
        raise ValueError(f"Unable to parse requirement: {requirement!r}")
    return match.group(0)


def main():
    print("[Lifecycle Bootstrap] Inspecting environment dependencies...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_env_dir = os.path.abspath(os.path.join(current_dir, "..", "python_embedded"))
    requirements_file = os.path.join(current_dir, "requirements.txt")

    os.makedirs(target_env_dir, exist_ok=True)

    if os.path.exists(requirements_file):
        with open(requirements_file) as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        required_packages = ["torch>=2.8.0", "pandas>=2.3.0", "onnx>=1.19.0", "zarr>=2.18,<3", "numcodecs>=0.15.0", "numpy>=1.26.0", "onnxscript>=0.1.0"]

    sys.path.insert(0, target_env_dir)

    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg_import_name(pkg))
        except ImportError:
            missing_packages.append(pkg)

    if not missing_packages:
        print("[Lifecycle Bootstrap] All dependencies are ready to run.")
        return

    print(f"[Lifecycle Bootstrap] Found missing dependencies: {missing_packages}. Initializing compilation...")

    for pkg in missing_packages:
        print(f"[Lifecycle Bootstrap] Compiling {pkg} into local isolated runtime...")
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                target_env_dir,
                pkg,
            ])
        except Exception as e:
            print(f"[Lifecycle Bootstrap] Critical Error during dependency mapping: {str(e)}")
            sys.exit(1)

    print("[Lifecycle Bootstrap] Local dependency mapping successfully established.")


if __name__ == "__main__":
    main()