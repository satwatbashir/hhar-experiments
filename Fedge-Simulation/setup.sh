#!/bin/bash
# Fedge-Simulation Setup Script
# This script creates a virtual environment and installs all dependencies

set -e

echo "=============================================="
echo "Fedge-Simulation Setup"
echo "=============================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."

# For CPU-only (faster, smaller)
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Or for GPU support (slower to download, requires CUDA)
# pip install torch

# Install other dependencies
pip install toml numpy pandas scipy scikit-learn matplotlib seaborn psutil filelock requests

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To run the simulation:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the simulation:"
echo "     cd fedge"
echo "     SEED=42 python orchestrator.py"
echo ""
echo "To change the number of rounds, edit fedge/pyproject.toml:"
echo "  global_rounds = 100  # Change this value"
echo ""
