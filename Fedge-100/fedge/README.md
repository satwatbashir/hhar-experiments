# Fedge: Hierarchical Federated Learning on HHAR

Hierarchical Federated Learning with SCAFFOLD and cloud-level clustering on the HHAR (Heterogeneity Human Activity Recognition) dataset.

## Project Structure

```
fedge/
├── fedge/
│   ├── task.py           # HHAR dataset processing and CNN model
│   ├── partitioning.py   # User-based data partitioning
│   ├── leaf_server.py    # Leaf server implementation
│   ├── cloud_server.py   # Cloud server with clustering
│   └── utils/            # Utility functions
├── orchestrator.py       # Main entry point
└── pyproject.toml        # Configuration
```

## Setup

### Google Cloud VM Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fedge.git
cd fedge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Dataset

The HHAR dataset is automatically downloaded on first run. It will be placed in `hhar/Activity recognition exp/`.

## Running Experiments

### Fedge (This Project)

```bash
source .venv/bin/activate
cd fedge

# Seed 1 (default)
SEED=42 python3 orchestrator.py

# Seed 2
SEED=43 python3 orchestrator.py

# Seed 3
SEED=44 python3 orchestrator.py
```

### FedProx

```bash
source .venv/bin/activate
cd fedprox

# Run with different seeds
SEED=42 flwr run .
SEED=43 flwr run .
SEED=44 flwr run .
```

### HierFL

```bash
source .venv/bin/activate
cd HierFL/fedge

# Run with different seeds
SEED=42 python3 orchestrator.py
SEED=43 python3 orchestrator.py
SEED=44 python3 orchestrator.py
```

## Configuration

Key parameters in `pyproject.toml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global_rounds` | Number of global rounds | 100 |
| `num_servers` | Number of leaf servers | 3 |
| `clients_per_server` | Clients per leaf server | [3, 3, 3] |
| `local_epochs` | Local training epochs | 5 |
| `seed` | Random seed | 42 |

## Results

Results are saved in:
- `metrics/cloud/rounds.csv` - Cloud-level metrics
- `metrics/leaf/server_X/` - Leaf server metrics

### Saving Results

```bash
# Create results folder
mkdir -p results_seed1
cp -r metrics results_seed1/

# Zip for download
zip -r results_seed1.zip results_seed1/
```

## Architecture

- **9 Clients**: One per HHAR user (natural non-IID distribution)
- **3 Leaf Servers**: Each manages 3 clients
- **1 Cloud Server**: Aggregates leaf server models with clustering

## Hardware

Experiments run on Google Cloud Platform:
- **VM Type**: e2-standard-8 (8 vCPU, 32 GB RAM)
- **Runtime**: ~8-10 hours per seed (HierFL/Fedge), ~4-5 hours (FedProx)
