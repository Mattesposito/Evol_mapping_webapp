import subprocess
import sys
from pathlib import Path
import importlib

def import_install_comet():
    # Courtesy of chatGPT
    repo_path = Path(__file__).parent / "comet-emu"
    setup_py = repo_path / "setup.py"

    # Step 1: If setup.py is missing, try initializing submodules
    if not setup_py.exists():
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=Path(__file__).parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Initialized submodules.")
        except subprocess.CalledProcessError as e:
            print("Failed to init submodules:", e.stderr.decode())
            raise RuntimeError("Submodule init failed.")

    # Step 2: Try importing comet
    try:
        from comet import comet
    except ImportError:
        # Step 3: Try installing comet from the submodule path
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(repo_path)],
                check=True
            )
            print("Installed comet in user mode.")
        except subprocess.CalledProcessError as e:
            print("Failed to install comet:", e)
            raise RuntimeError("Could not install comet.")

        # Step 4: Retry import
        try:
            importlib.invalidate_caches()
            from comet import comet
        except ImportError:
            raise ImportError("Failed to import comet even after installation.")
        
    return comet

# Call this early in your Streamlit script
# import_install_comet()
