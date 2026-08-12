import os
import sys
import subprocess

def main():
    print("[Lifecycle Bootstrap] Inspecting environment dependencies...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_env_dir = os.path.abspath(os.path.join(current_dir, "..", "python_embedded"))
    requirements_file = os.path.join(current_dir, "requirements.txt")
    
    os.makedirs(target_env_dir, exist_ok=True)
    
    # Read packages from requirements.txt
    if os.path.exists(requirements_file):
        with open(requirements_file) as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        required_packages = required_packages = ["torch>=2.8.0", "pandas>=2.3.0", "onnx>=1.19.0"]
    
    # We add target_env_dir to sys.path during verification to see if they are already installed locally
    sys.path.insert(0, target_env_dir)
    
    missing_packages = []
    for pkg in required_packages:
        pkg_name = pkg.split("==")[0]  # Handle "torch==2.8.0"
        try:
            __import__(pkg_name)
        except ImportError:
            missing_packages.append(pkg)
            
    if not missing_packages:
        print("[Lifecycle Bootstrap] All dependencies are ready to run.")
        return

    print(f"[Lifecycle Bootstrap] Found missing dependencies: {missing_packages}. Initializing compilation...")
    
    # Execute pip install targeted straight into your isolated python_embedded directory
    for pkg in missing_packages:
        print(f"[Lifecycle Bootstrap] Compiling {pkg} into local isolated runtime...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--target", target_env_dir, pkg
            ])
        except Exception as e:
            print(f"[Lifecycle Bootstrap] Critical Error during dependency mapping: {str(e)}")
            sys.exit(1)
            
    print("[Lifecycle Bootstrap] Local dependency mapping successfully established.")

if __name__ == "__main__":
    main()