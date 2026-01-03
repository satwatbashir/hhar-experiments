# HHAR Federated Learning Experiments

Federated Learning experiments on the HHAR (Heterogeneous Human Activity Recognition) dataset, comparing multiple FL methods under natural non-IID data distribution.

## Methods Implemented

| Method | Description | Run Command |
|--------|-------------|-------------|
| **FedProx** | Proximal term regularization | `cd fedprox && SEED=42 flwr run .` |
| **HierFL** | Hierarchical FL (no clustering) | `cd HierFL/fedge && SEED=42 python3 orchestrator.py` |
| **Fedge** | Hierarchical FL with cloud clustering | `cd Fedge-100/fedge && SEED=42 python3 orchestrator.py` |
| **SCAFFOLD** | Control variates for variance reduction | `cd Scaffold && flwr run .` |
| **CFL** | Clustered Federated Learning | `cd CFL && flwr run .` |
| **pFedMe** | Personalized FL with Moreau Envelopes | `cd pFedMe && flwr run .` |

## Dataset

- **HHAR Dataset** from UCI Machine Learning Repository
- 9 users with smartphones/smartwatches
- 6 activity classes: Walking, Sitting, Standing, Biking, Stairs Up, Stairs Down
- **Non-IID partitioning**: User-based (each user = 1 client)

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/satwatbashir/hhar-experiments.git
cd hhar-experiments
```

### 2. Setup environment (Google Cloud VM with GPU)
```bash
chmod +x setup_env.sh
./setup_env.sh
source ~/.venv/bin/activate
```

### 3. Download HHAR dataset
```bash
# Download to each method's folder
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip
unzip "Activity recognition exp.zip"

# Copy to each method (example for fedprox)
mkdir -p fedprox/hhar
cp -r "Activity recognition exp" fedprox/hhar/
```

### 4. Run experiments

```bash
# Activate virtual environment
source ~/.venv/bin/activate

# FedProx (100 rounds)
cd fedprox
SEED=42 flwr run .   # Seed 1
SEED=43 flwr run .   # Seed 2
SEED=44 flwr run .   # Seed 3

# HierFL (100 rounds, hierarchical)
cd HierFL/fedge
SEED=42 python3 orchestrator.py   # Seed 1
SEED=43 python3 orchestrator.py   # Seed 2
SEED=44 python3 orchestrator.py   # Seed 3

# Fedge (100 rounds, hierarchical with clustering)
cd Fedge-100/fedge
SEED=42 python3 orchestrator.py   # Seed 1
SEED=43 python3 orchestrator.py   # Seed 2
SEED=44 python3 orchestrator.py   # Seed 3

# Use screen for long runs
screen -S experiment
SEED=42 flwr run .
# Ctrl+A, D to detach
# screen -r experiment to reattach
```

## Configuration

All methods use consistent settings for fair comparison:

| Parameter | Value |
|-----------|-------|
| Rounds | 100 |
| Clients | 9 (user-based) |
| Local Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 0.05 |
| Fraction Fit | 1.0 (100% participation) |

Method-specific parameters:
- **FedProx**: `proximal_mu = 0.01`
- **pFedMe**: `lambda = 15.0`, `inner_lr = 0.05`, `outer_lr = 0.01`
- **HierFL**: 3 leaf servers, 3 clients each

## Metrics

Each method generates metrics in the `metrics/` folder:

- `centralized_metrics.csv` - Global model evaluation per round
- `clients.csv` - Per-client metrics per round
- `rounds.csv` - Aggregated metrics with 95% CI

### Key metrics tracked:
- Test/Train Accuracy and Loss
- Accuracy Gap (train - test)
- Convergence rate and stability
- Communication costs (bytes up/down)
- Computation time

## Saving and Downloading Results

### FedProx
```bash
cd ~/hhar/fedprox
mkdir -p results_seed1
mv metrics results_seed1/
zip -r fedprox_seed1.zip results_seed1/
```
Download path: `/home/satwatbashir/hhar/fedprox/fedprox_seed1.zip`

### HierFL / Fedge
```bash
cd ~/hhar/HierFL/fedge  # or ~/hhar/Fedge-100/fedge
mkdir -p results_seed1/metrics/cloud
mkdir -p results_seed1/metrics/leaf/server_0
mkdir -p results_seed1/metrics/leaf/server_1
mkdir -p results_seed1/metrics/leaf/server_2

# Copy cloud metrics
cp metrics/cloud/rounds.csv results_seed1/metrics/cloud/

# Copy leaf server metrics
cp -r rounds/leaf/server_0/metrics/* results_seed1/metrics/leaf/server_0/
cp -r rounds/leaf/server_1/metrics/* results_seed1/metrics/leaf/server_1/
cp -r rounds/leaf/server_2/metrics/* results_seed1/metrics/leaf/server_2/

# Zip
zip -r hierfl_seed1.zip results_seed1/
```
Download path: `/home/satwatbashir/hhar/HierFL/fedge/hierfl_seed1.zip`

### Cleanup before next run
```bash
# FedProx
rm -rf metrics

# HierFL / Fedge
rm -rf rounds runs signals metrics
```

## Reproducibility

All experiments support multiple seeds via environment variable:

```bash
SEED=42 python3 orchestrator.py   # Seed 1
SEED=43 python3 orchestrator.py   # Seed 2
SEED=44 python3 orchestrator.py   # Seed 3
```

Seeds control: random, numpy, and PyTorch random number generators.

## Hardware

Experiments run on Google Cloud Platform:
- **VM Type**: e2-standard-8 (8 vCPU, 32 GB RAM)
- **Runtime**: ~8-10 hours per seed (HierFL/Fedge), ~4-5 hours (FedProx)

## Requirements

- Python 3.10+
- PyTorch 2.x
- Flower 1.18+

See `setup_env.sh` for complete dependency installation.

## Project Structure

```
HHAR/
├── setup_env.sh          # Environment setup script
├── README.md             # This file
├── fedprox/              # FedProx implementation
│   ├── fedge/
│   │   ├── client_app.py
│   │   ├── server_app.py
│   │   └── task.py
│   ├── pyproject.toml
│   └── metrics/
├── HierFL/               # Hierarchical FL (no clustering)
│   └── fedge/
│       ├── orchestrator.py
│       ├── cloud_server.py
│       ├── leaf_server.py
│       └── task.py
├── Fedge-100/            # Hierarchical FL with clustering
│   └── fedge/
│       ├── orchestrator.py
│       ├── cloud_server.py
│       ├── leaf_server.py
│       └── task.py
├── Scaffold/             # SCAFFOLD
├── CFL/                  # Clustered FL
└── pFedMe/               # Personalized FedMe
```

## Citation

If you use this code, please cite:
- HHAR Dataset: Stisen et al., "Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition" (SenSys 2015)
- Flower Framework: Beutel et al., "Flower: A Friendly Federated Learning Framework" (2020)
