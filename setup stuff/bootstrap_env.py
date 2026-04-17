#!/usr/bin/env python3
"""
Auto-Setup Script for Actuator Net
Cross-platform setup script for Windows, macOS, and Linux
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path

def run_command(cmd, check=True, capture_output=True):
    """Run a command and handle errors"""
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
        if result.stdout and capture_output:
            print(result.stdout)
        if result.stderr and capture_output:
            print(result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False

def check_conda():
    """Check if conda is available"""
    print("Checking for conda...")
    if not run_command(["conda", "--version"], check=False, capture_output=False):
        print("ERROR: Conda not found. Please install Miniforge first:")
        print("   https://github.com/conda-forge/miniforge")
        print("   Then restart your terminal and run this script again.")
        return False
    print("SUCCESS: Conda found")
    return True

def detect_os():
    """Detect the operating system"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"

def detect_nvidia_gpu():
    """Detect if NVIDIA GPU is available"""
    print("Checking for NVIDIA GPU...")
    
    # Try nvidia-smi first
    if run_command(["nvidia-smi"], check=False, capture_output=False):
        print("NVIDIA GPU detected via nvidia-smi")
        return True
    
    # On Linux, try lspci
    if platform.system().lower() == "linux":
        if run_command(["lspci"], check=False, capture_output=False):
            result = subprocess.run(["lspci"], capture_output=True, text=True)
            if "nvidia" in result.stdout.lower():
                print("NVIDIA GPU detected via lspci")
                return True
    
    print("No NVIDIA GPU detected, using CPU-only PyTorch")
    return False

def create_environment(has_nvidia_gpu):
    """Create the conda environment"""
    if has_nvidia_gpu:
        print("Creating conda environment with CUDA support...")
        env_file = "environment-gpu.yml"
    else:
        print("Creating conda environment...")
        env_file = "environment.yml"
    
    if not os.path.exists(env_file):
        print(f"ERROR: Environment file {env_file} not found")
        return False
    
    # Try to create the environment, and remove if it already exists
    print("Creating conda environment...")
    result = run_command(["conda", "env", "create", "-f", env_file, "-n", "actuator-net"])
    
    # If creation failed because environment exists, remove and recreate
    if not result:
        print("Environment creation failed. Checking if environment already exists...")
        env_check = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        if "actuator-net" in env_check.stdout:
            print("Environment 'actuator-net' already exists. Removing it first...")
            if run_command(["conda", "env", "remove", "-n", "actuator-net", "-y"]):
                print("Recreating environment...")
                return run_command(["conda", "env", "create", "-f", env_file, "-n", "actuator-net"])
            else:
                print("ERROR: Failed to remove existing environment")
                return False
        else:
            print("ERROR: Environment creation failed for unknown reason")
            return False
    
    return True

def install_pytorch(os_name, has_nvidia_gpu):
    """Install PyTorch with appropriate CUDA support"""
    print("Installing PyTorch...")
    
    if os_name == "macos":
        print("   Installing PyTorch for macOS...")
        cmd = ["conda", "run", "-n", "actuator-net", "pip", "install", 
               "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1"]
    elif has_nvidia_gpu:
        print("   Installing PyTorch with CUDA support...")
        cmd = ["conda", "run", "-n", "actuator-net", "pip", "install",
               "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
               "--index-url", "https://download.pytorch.org/whl/cu128"]
    else:
        print("   Installing PyTorch CPU-only version...")
        cmd = ["conda", "run", "-n", "actuator-net", "pip", "install",
               "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
               "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    return run_command(cmd)

def install_ml_packages():
    """Install ML and data science packages"""
    print("Installing ML and Data Science libraries...")
    
    # First check if the environment exists and is working
    print("Verifying environment is working...")
    if not run_command(["conda", "run", "-n", "actuator-net", "python", "-c", "print('Environment is working')"]):
        print("ERROR: Environment 'actuator-net' is not working properly")
        return False
    
    # Install the packages
    return run_command(["conda", "run", "-n", "actuator-net", "pip", "install", "."])

def test_setup():
    """Test that the setup works"""
    print("Testing the setup...")
    
    test_commands = [
        ["conda", "run", "-n", "actuator-net", "python", "-c", 
         "import torch; print('SUCCESS: PyTorch', torch.__version__, 'installed successfully')"],
        ["conda", "run", "-n", "actuator-net", "python", "-c", 
         "import numpy; print('SUCCESS: NumPy', numpy.__version__, 'installed successfully')"],
        ["conda", "run", "-n", "actuator-net", "python", "-c", 
         "import matplotlib; print('SUCCESS: Matplotlib', matplotlib.__version__, 'installed successfully')"],
        ["conda", "run", "-n", "actuator-net", "python", "-c", 
         "import sklearn; print('SUCCESS: Scikit-learn', sklearn.__version__, 'installed successfully')"]
    ]
    
    for cmd in test_commands:
        if not run_command(cmd):
            print(f"WARNING: Test failed for {cmd[-1]}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("Setting up Actuator Net Environment...")
    print("=" * 50)
    
    # Detect OS
    os_name = detect_os()
    print(f"Detected OS: {os_name}")
    
    # Check if conda is available
    if not check_conda():
        return 1
    
    # Detect NVIDIA GPU
    has_nvidia_gpu = detect_nvidia_gpu()
    
    # Create conda environment
    if not create_environment(has_nvidia_gpu):
        print("ERROR: Failed to create conda environment")
        return 1
    
    # Install PyTorch
    if not install_pytorch(os_name, has_nvidia_gpu):
        print("ERROR: Failed to install PyTorch")
        return 1
    
    # Install ML packages
    if not install_ml_packages():
        print("ERROR: Failed to install ML packages")
        return 1
    
    # Test the setup
    if not test_setup():
        print("WARNING: Some tests failed, but setup may still work")
    
    print("\nSetup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Open VS Code")
    print("2. Open this folder in VS Code")
    print("3. Click on any .ipynb file in the 'notebooks' folder")
    print("4. Select 'actuator-net' kernel when prompted")
    print("5. Start coding!")
    print("\nUseful commands:")
    print("  conda activate actuator-net    # Activate environment")
    print("  conda deactivate                   # Deactivate environment")
    print("\nHappy learning!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())