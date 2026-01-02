# HHAR Federated Learning Experiments Plan

## Methods to Compare
1. **FedProx** - Proximal regularization baseline
2. **HierFL** - Hierarchical FL with cloud clustering
3. **Fedge** - Your proposed method (hierarchical + SCAFFOLD + dynamic clustering)

---

## Fedge Code Verification (READY TO RUN)

Verified that Fedge handles parameters correctly - no bugs like FedProx had. Key checks:
- `cloud_flower.py`: Properly uses `parameters_to_ndarrays()` with None checks
- `leaf_server.py`: Correct parameter handling in SCAFFOLD aggregation
- `orchestrator.py`: Signal files and model paths work correctly

**To change rounds to 200:** Edit `Fedge-100/fedge/pyproject.toml` line 33:
```toml
global_rounds = 200
```

---

## Updated Experiment Plan (200 Rounds, 3 Seeds)

### Phase 1: Baselines (FedProx + HierFL)
| Method | Rounds | Seeds | Status |
|--------|--------|-------|--------|
| FedProx | 200 | 3 | Seed 1 done (100 rounds), need seeds 2-3 at 200 |
| HierFL | 200 | 3 | Seed 1 done (100 rounds), need seeds 2-3 at 200 |

### Phase 2: Your Method (Fedge)
| Method | Rounds | Seeds | Status |
|--------|--------|-------|--------|
| Fedge | 200 | 3 | Ready to run after baselines |

**Total runs needed:** 9 runs (3 methods x 3 seeds)

---

## Priority Order

### Must Have (for publication):
1. **FedProx seed 2** - 200 rounds
2. **FedProx seed 3** - 200 rounds
3. **HierFL seed 2** - 200 rounds
4. **HierFL seed 3** - 200 rounds
5. **Fedge seed 1** - 200 rounds
6. **Fedge seed 2** - 200 rounds
7. **Fedge seed 3** - 200 rounds

### Why 3 Seeds?
- Sufficient for mean +/- std deviation
- Enables 95% confidence intervals
- Standard for ML conferences (NeurIPS, ICML, etc.)
- Some journals prefer 5 seeds, but 3 is acceptable

---

## Settings Summary

### Common Settings (All Methods)
| Parameter | Value |
|-----------|-------|
| Clients | 9 (user-based partitioning) |
| Local Epochs | 5 |
| Batch Size | 32 (FedProx), 64 (HierFL/Fedge) |
| Learning Rate | 0.05 |
| Dataset | HHAR (Activity recognition exp) |
| Train/Test Split | 80/20 |

### FedProx Specific
| Parameter | Value |
|-----------|-------|
| proximal_mu | 0.01 |

### HierFL Specific
| Parameter | Value |
|-----------|-------|
| Leaf Servers | 3 |
| Clients per Server | [3, 3, 3] |
| Server Rounds per Global | 1 |

### Fedge Specific (Your Method)
| Parameter | Value |
|-----------|-------|
| Leaf Servers | 3 |
| Clients per Server | [3, 3, 3] |
| SCAFFOLD | Enabled (server-side) |
| FedProx | Enabled (prox_mu=0.001) |
| Cloud Clustering | Enabled (tau=0.7) |
| Cluster Method | cosine_similarity |

---

## Folder Structure for Results

```
~/hhar/
├── fedprox/
│   ├── results_seed1/    # Done (100 rounds)
│   ├── results_seed2/    # To run (200 rounds)
│   └── results_seed3/    # To run (200 rounds)
│
├── HierFL/fedge/
│   ├── results_seed1/    # Done (100 rounds)
│   ├── results_seed2/    # To run (200 rounds)
│   └── results_seed3/    # To run (200 rounds)
│
└── Fedge-100/fedge/
    ├── results_seed1/    # To run (200 rounds)
    ├── results_seed2/    # To run (200 rounds)
    └── results_seed3/    # To run (200 rounds)
```

---

## Commands Summary

### FedProx (200 rounds):
```bash
cd ~/hhar/fedprox
# First, update pyproject.toml: num-server-rounds = 200
flwr run .
# After: mkdir results_seed2 && mv metrics results_seed2/
```

### HierFL (200 rounds):
```bash
cd ~/hhar/HierFL/fedge
# First, update pyproject.toml: global_rounds = 200
rm -rf signals models
python3 orchestrator.py
# After: mkdir results_seed2 && mv metrics rounds runs results_seed2/
```

### Fedge (200 rounds):
```bash
cd ~/hhar/Fedge-100/fedge
# First, update pyproject.toml: global_rounds = 200
rm -rf signals models rounds
python3 orchestrator.py
# After: mkdir results_seed1 && mv metrics rounds runs clusters results_seed1/
```

---

## Expected Results Table (for paper)

| Method | Test Accuracy (mean +/- std) | Test Loss | Key Features |
|--------|------------------------------|-----------|--------------|
| FedProx | 50.6% +/- ? | 2.48 +/- ? | Proximal regularization |
| HierFL | 63.4% +/- ? | 1.69 +/- ? | Hierarchical + clustering |
| Fedge | TBD | TBD | Hier + SCAFFOLD + dynamic clustering |

*Need 3 seeds to calculate std and confidence intervals*

---

## Estimated Total Time

| Method | Runs | Time per Run | Total |
|--------|------|--------------|-------|
| FedProx (200 rounds) | 2 | ~6-8 hrs | ~12-16 hrs |
| HierFL (200 rounds) | 2 | ~16-20 hrs | ~32-40 hrs |
| Fedge (200 rounds) | 3 | ~16-20 hrs | ~48-60 hrs |

**Total for all experiments:** ~90-120 hours (spread over multiple days)

---

## File Locations Reference

| File | Purpose |
|------|---------|
| `/mnt/d/learn/HHAR/fedprox/pyproject.toml` | FedProx config |
| `/mnt/d/learn/HHAR/HierFL/fedge/pyproject.toml` | HierFL config |
| `/mnt/d/learn/HHAR/Fedge-100/fedge/pyproject.toml` | Fedge config |
| `metrics/cloud/rounds.csv` | Global metrics per round |
| `rounds/leaf/server_X/metrics/` | Per-server metrics |
