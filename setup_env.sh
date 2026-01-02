#!/bin/bash
# HHAR Federated Learning - Environment Setup Script
# Run this script on a fresh Ubuntu VM (tested on Ubuntu 22.04 LTS)

set -e  # Exit on any error

echo "=== HHAR Environment Setup ==="
echo ""

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git unzip wget

# 2. Create virtual environment
echo "[2/5] Creating virtual environment..."
cd ~
python3 -m venv .venv
source ~/.venv/bin/activate

# 3. Install PyTorch with CUDA support
echo "[3/5] Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4. Install Flower and simulation dependencies
echo "[4/5] Installing Flower and dependencies..."
pip install -U "flwr[simulation]"
pip install flwr-datasets

# 5. Install other required packages
echo "[5/5] Installing additional packages..."
pip install pandas numpy scipy matplotlib seaborn tqdm typing-extensions toml

# Verify installations
echo ""
echo "=== Verifying Installation ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import flwr; print(f'Flower: {flwr.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source ~/.venv/bin/activate"
echo ""
echo "To run experiments:"
echo "  FedProx:  cd ~/hhar-experiments/fedprox && flwr run ."
echo "  HierFL:   cd ~/hhar-experiments/HierFL/fedge && python3 orchestrator.py"
echo "  Scaffold: cd ~/hhar-experiments/Scaffold && flwr run ."
echo "  CFL:      cd ~/hhar-experiments/CFL && flwr run ."
echo "  pFedMe:   cd ~/hhar-experiments/pFedMe && flwr run ."
echo ""
