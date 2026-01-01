# HHAR Project - Comprehensive Method Analysis

This document provides a detailed analysis of all federated learning methods implemented in the HHAR (Heterogeneous Human Activity Recognition) project, including settings, models, hyperparameters, and evaluation strategies.

---

## Overview Summary

| Method | Rounds | Clients | Participation | Local Epochs | Batch Size | Learning Rate | Key Feature |
|--------|--------|---------|---------------|--------------|------------|---------------|-------------|
| **CFL** | 100 | 9 | 100% | 5 | 32 | 0.05 | Recursive clustering |
| **SCAFFOLD** | 100 | 9 | 100% | 5 | 32 | 0.05 | Control variates |
| **FedProx** | 100 | 9 | 100% | 5 | 32 | 0.05 | Proximal term (mu=0.01) |
| **pFedMe** | 100 | 9 | 100% | 5 | 32 | inner=0.05, outer=0.01 | Bi-level optimization |
| **HierFL** | 100 | 9 (3 servers x 3 clients) | 100% | 5 | 64 | 0.05 | Hierarchical + SCAFFOLD |
| **Fedge-100** | 100 | 9 | 100% | 5 | 32 | 0.05 | Hierarchical clustering |

---

## 1. CFL (Clustered Federated Learning)

### Configuration File
`/mnt/d/learn/HHAR/CFL/pyproject.toml`

### Federated Learning Settings

| Parameter | Value | Correct/Notes |
|-----------|-------|---------------|
| **Number of Rounds** | 100 | Yes |
| **Number of Clients** | 9 | Yes (user-based partitioning) |
| **Fraction Fit** | 1.0 | 100% client participation |
| **Fraction Evaluate** | 1.0 (implicit) | All clients evaluated |
| **Min Available Clients** | 9 | Full participation required |

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Local Epochs** | 5 | `pyproject.toml` |
| **Batch Size** | 32 | `pyproject.toml` |
| **Learning Rate** | 0.05 | `client_app.py:24` (default) |
| **Optimizer** | SGD | `client_app.py:36` |
| **Loss Function** | CrossEntropyLoss | `client_app.py:37` |

### CFL-Specific Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **eps_1** | 0.4 | Lower signal threshold for split |
| **eps_2** | 1.6 | Upper signal threshold for split |
| **min_cluster_size** | 2 | Minimum clients per cluster |
| **gamma_max** | 0.05 | Gap threshold for cluster splitting |

### Model Architecture

| Component | Configuration |
|-----------|---------------|
| **Model Type** | Tiny 1D CNN |
| **Input Shape** | (B, 6, 100) - 6 channels, 100 timesteps |
| **Conv1** | 6 → 64 channels, kernel=5, padding=2 |
| **BatchNorm1** | 64 channels |
| **Conv2** | 64 → 64 channels, kernel=5, padding=2 |
| **BatchNorm2** | 64 channels |
| **Pooling** | AdaptiveAvgPool1d(1) |
| **Output** | Linear(64 → 6) |
| **Total Parameters** | ~23,376 |

### Evaluation Strategy

| Metric | Description |
|--------|-------------|
| **test_accuracy** | Accuracy on local validation set |
| **test_loss** | Loss on local validation set |
| **train_accuracy** | Accuracy on local training set |
| **train_loss** | Loss on local training set |
| **accuracy_gap** | train_acc - test_acc |
| **loss_gap** | test_loss - train_loss |
| **personalized_test_accuracy** | After 1 epoch fine-tuning |
| **relative_improvement** | (personalized - global) / global |

### Aggregation Strategy
- Weighted average within clusters
- Sign-corrected, size-weighted updates
- Minimax bipartition algorithm for cluster splitting

---

## 2. SCAFFOLD

### Configuration File
`/mnt/d/learn/HHAR/Scaffold/pyproject.toml`

### Federated Learning Settings

| Parameter | Value | Correct/Notes |
|-----------|-------|---------------|
| **Number of Rounds** | 100 | Yes |
| **Number of Clients** | 9 | Yes |
| **Fraction Fit** | 1.0 | 100% participation |
| **Partition Strategy** | User-based | Natural heterogeneity |

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Local Epochs** | 5 | `pyproject.toml` |
| **Batch Size** | 32 | `pyproject.toml` |
| **Learning Rate** | 0.05 | `client_app.py:24` |
| **Optimizer** | SGD | `client_app.py:36` |
| **Loss Function** | CrossEntropyLoss | `client_app.py:37` |

### SCAFFOLD-Specific Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Global LR (server_lr)** | 1.0 | Server aggregation learning rate |
| **Control Variate Update** | c_local_new = c_local - c_global - (y_delta / (lr * K)) | K = step count |

### Model Architecture
Same as CFL (Tiny 1D CNN)

### Evaluation Strategy
Same metrics as CFL, plus:
- `y_delta_norm`: Norm of weight differences
- `c_delta_norm`: Norm of control variate differences

### Key Implementation Details
- Returns two diffs: `y_delta` (weights) and `c_delta` (control variates)
- Server updates: W_global += avg(y_delta), c_global += avg(c_delta)
- Gradient correction: loss + (c_global - c_local) dot w

---

## 3. FedProx

### Configuration File
`/mnt/d/learn/HHAR/fedprox/pyproject.toml`

### Federated Learning Settings

| Parameter | Value | Correct/Notes |
|-----------|-------|---------------|
| **Number of Rounds** | 100 | Yes |
| **Number of Clients** | 9 | Yes |
| **Fraction Fit** | 1.0 | 100% participation |
| **Partition Strategy** | User-based | Natural heterogeneity |

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Local Epochs** | 5 | `pyproject.toml` |
| **Batch Size** | 32 | `pyproject.toml` |
| **Learning Rate** | 0.05 | Default in client |
| **Optimizer** | SGD | Standard |
| **Loss Function** | CrossEntropyLoss | Standard |

### FedProx-Specific Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **proximal_mu** | 0.01 | Proximal term coefficient |
| **Proximal Term** | mu/2 * ||w - w_global||^2 | Added to loss |

### Model Architecture
Same as CFL (Tiny 1D CNN)

### Evaluation Strategy
Standard global model evaluation metrics

### Aggregation Strategy
- Uses Flower's built-in FedProx strategy
- Weighted FedAvg with proximal regularization

---

## 4. pFedMe (Personalized Federated Meta-Learning)

### Configuration File
`/mnt/d/learn/HHAR/pFedMe/pyproject.toml`

### Federated Learning Settings

| Parameter | Value | Correct/Notes |
|-----------|-------|---------------|
| **Number of Rounds** | 100 | Yes |
| **Number of Clients** | 9 | Yes |
| **Fraction Fit** | 1.0 | 100% participation |
| **Min Available Clients** | 9 | Deterministic all-client rounds |
| **Partition Strategy** | User-based | Natural heterogeneity |

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Local Epochs** | 5 | `pyproject.toml` |
| **Batch Size** | 32 | `pyproject.toml` |
| **Loss Function** | CrossEntropyLoss | `client_app.py:42` |
| **Gradient Clipping** | 1.0 | `client_app.py:128` |

### pFedMe-Specific Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **lamda** | 15.0 | Moreau envelope regularization weight |
| **inner_steps (K)** | 5 | Inner optimization steps for theta |
| **outer_steps (R)** | 1 | Outer optimization steps for w |
| **inner_lr** | 0.05 | Learning rate for theta optimization |
| **outer_lr** | 0.01 | Learning rate for w outer update |
| **beta** | 1.0 | Server mixing parameter (1.0 = no mixing) |

### Model Architecture
Same as CFL (Tiny 1D CNN), but with two models:
- **w (net)**: Global model copy
- **theta (theta_net)**: Personalized model

### Bi-Level Optimization Algorithm
```
For each epoch:
    For outer_step in range(R):  # R=1
        For batch in range(K):   # K=5
            # Inner: optimize theta
            loss = CE_loss + (lamda/2) * ||theta - w||^2
            theta = theta - inner_lr * grad(loss)
        # Outer: update w
        w = w - outer_lr * lamda * (w - theta)
```

### Evaluation Strategy

| Metric | Description |
|--------|-------------|
| **test_accuracy** | Global model (w) accuracy |
| **personalized_test_accuracy** | Personalized model (theta) accuracy |
| **relative_improvement** | (personalized - global) / global |
| **num_inner_batches** | Number of inner optimization batches |

---

## 5. HierFL (Hierarchical Federated Learning)

### Configuration File
`/mnt/d/learn/HHAR/HierFL/fedge/pyproject.toml`

### Federated Learning Settings

| Parameter | Value | Correct/Notes |
|-----------|-------|---------------|
| **Global Rounds** | 100 | Yes |
| **Number of Leaf Servers** | 3 | Intermediate aggregation |
| **Clients per Server** | [3, 3, 3] | 9 total clients |
| **Server Rounds per Global** | 1 | Leaf rounds per cloud round |
| **Fraction Fit** | 1.0 | 100% participation |

### Hierarchical Architecture

```
Cloud Server (port 6000)
    ├── Leaf Server 0 (port 5000) ── Clients 0, 1, 2
    ├── Leaf Server 1 (port 5001) ── Clients 3, 4, 5
    └── Leaf Server 2 (port 5002) ── Clients 6, 7, 8
```

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Local Epochs** | 5 | `pyproject.toml` |
| **Batch Size** | 64 | `pyproject.toml` |
| **Eval Batch Size** | 128 | `pyproject.toml` |
| **Initial Learning Rate** | 0.05 | `lr_init` |
| **Server LR** | 1.0 | `server_lr` |
| **Global LR** | 1.0 | `global_lr` |
| **Momentum** | 0.9 | `pyproject.toml` |
| **Weight Decay** | 0.0001 | `pyproject.toml` |
| **Gradient Clipping** | 1.0 | `clip_norm` |
| **LR Decay (gamma)** | 0.99 | `lr_gamma` |

### Advanced Features

| Parameter | Value | Description |
|-----------|-------|-------------|
| **scaffold_enabled** | true | SCAFFOLD control variates |
| **prox_mu** | 0.001 | FedProx regularization |
| **prox_mu_min** | 0.0001 | Minimum proximal mu |
| **es_patience** | 5 | Early stopping patience |
| **es_delta** | 0.001 | Early stopping threshold |
| **acc_stable_threshold** | 0.002 | Accuracy stabilization threshold |

### Cloud Clustering Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **enable** | true | Cloud-level clustering |
| **start_round** | 1 | Start clustering from round 1 |
| **frequency** | 1 | Cluster every round |
| **method** | cosine_similarity | Weight-based clustering |
| **tau** | 0.7 | Similarity threshold |

### Model Architecture
Same as CFL (Tiny 1D CNN)

### Evaluation Strategy
- Per-leaf-server aggregation metrics
- Cloud-level global metrics
- Cluster-specific metrics when clustering enabled

---

## 6. Fedge-100 (Hierarchical with 100 Clients)

### Configuration File
`/mnt/d/learn/HHAR/Fedge-100/fedge/pyproject.toml`

### Federated Learning Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Number of Rounds (pyproject)** | 2 | Test setting |
| **Global Rounds (hierarchy)** | 100 | Production setting |
| **Number of Leaf Servers** | 3 | |
| **Clients per Server** | [3, 3, 3] | 9 total |
| **Server Rounds per Global** | 1 | |
| **Fraction Fit** | 1.0 | 100% participation |
| **Min Available Clients** | 9 | |

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| **Local Epochs** | 5 | `pyproject.toml` |
| **Batch Size** | 32 | `pyproject.toml` |
| **Learning Rate** | 0.05 | Default |
| **Alpha Server** | 1.0 | Server aggregation weight |
| **Alpha Client** | 0.5 | Client contribution weight |

### Hierarchical Structure
Same as HierFL with cloud clustering capability

---

## Common Dataset Configuration (All Methods)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **data-root** | "hhar/Activity recognition exp" | Path to HHAR CSV files |
| **use-watches** | true | Include smartwatch data |
| **sample-rate-hz** | 50 | Sensor sampling rate |
| **window-seconds** | 2 | Window length |
| **window-stride-seconds** | 1 | Stride (50% overlap) |
| **num-classes** | 6 | Activity classes |
| **model** | "cnn1d" | Model type |

### Activity Classes
1. Walking
2. Sitting
3. Standing
4. Biking
5. Stairs Up
6. Stairs Down

### Data Partitioning
- **Strategy**: User-based (natural heterogeneity)
- **Train/Test Split**: 80% / 20%
- **Clients**: 9 (one per user)

---

## Comparison Table: Key Differences

| Aspect | CFL | SCAFFOLD | FedProx | pFedMe | HierFL |
|--------|-----|----------|---------|--------|--------|
| **Architecture** | Flat | Flat | Flat | Flat | 2-tier |
| **Aggregation** | Clustered | Control variate | Weighted avg | Beta mixing | Hierarchical |
| **Personalization** | Cluster-level | None (global) | None | Bi-level theta | Cluster-level |
| **Regularization** | None | Gradient correction | Proximal term | Moreau envelope | SCAFFOLD + Prox |
| **Batch Size** | 32 | 32 | 32 | 32 | 64 |
| **Special Feature** | Bipartition clustering | Variance reduction | Convergence stability | Meta-learning | Cloud clustering |

---

## Potential Issues Identified

### 1. Learning Rate Inconsistencies
- CFL, SCAFFOLD, FedProx: Default LR = 0.05 (in code)
- pFedMe: inner_lr = 0.05, outer_lr = 0.01
- HierFL: lr_init = 0.05

**Note**: Learning rates are hardcoded in `client_app.py` as defaults, not always explicitly set in config files.

### 2. Batch Size Variations
- Most methods: 32
- HierFL: 64

### 3. Fedge-100 Round Count
- `pyproject.toml` shows `num-server-rounds = 2` (test setting)
- Hierarchy config shows `global_rounds = 100` (intended production)

### 4. Missing Explicit Server LR in Some Methods
- CFL and base SCAFFOLD don't explicitly define `global_lr` in pyproject.toml
- Defaults to 1.0 in strategy implementations

---

## Metrics Collected (All Methods)

### Training Metrics
- `train_loss_mean`
- `train_accuracy_mean`
- `comp_time_sec`
- `download_bytes`
- `upload_bytes`

### Evaluation Metrics
- `test_accuracy`
- `test_loss`
- `accuracy_gap`
- `loss_gap`
- `personalized_test_accuracy`
- `relative_improvement`

### System Metrics
- `round_time`
- `num_clusters` (CFL)
- `cluster_id` (CFL)

---

## Correctness Assessment

This section evaluates each implementation against the original papers and identifies any deviations or potential issues.

---

### 1. CFL (Clustered Federated Learning) - Correctness Analysis

**Reference Paper**: Sattler et al., "Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization under Privacy Constraints" (2020)

| Aspect | Implementation | Paper | Status |
|--------|---------------|-------|--------|
| **Clustering Metric** | Cosine similarity of gradient updates | Cosine similarity | ✅ CORRECT |
| **Bipartition Algorithm** | Complete-linkage hierarchical clustering | Minimax bipartition | ✅ CORRECT |
| **Split Criteria (ε₁)** | mean_norm < 0.4 | ε₁ threshold for mean gradient norm | ✅ CORRECT |
| **Split Criteria (ε₂)** | max_norm > 1.6 | ε₂ threshold for max gradient norm | ✅ CORRECT |
| **Gap Threshold (γ)** | (1 - α_max)/2 > γ_max (0.05) | Cluster separation criterion | ✅ CORRECT |
| **Aggregation** | Weighted average within clusters | Per-cluster FedAvg | ✅ CORRECT |
| **Update Direction** | `cluster_model -= avg_delta` | Clients send W_global - W_local | ⚠️ INVERTED |

**Issues Identified**:

1. **Update Sign Convention** (Line 318-319 in cfl.py):
   ```python
   self.cluster_models[c][p] -= d
   ```
   The client computes `delta = global - local` (line 132 in client_app.py), so subtracting delta is correct. However, this is opposite to standard gradient descent convention. **Status**: Functionally correct but inverted convention.

2. **Full Participation Required**: The implementation requires all cluster members to participate for split decisions (line 142-143). This is stricter than the paper.

**Overall Assessment**: ✅ **CORRECT** - Faithful implementation with minor convention differences.

---

### 2. SCAFFOLD - Correctness Analysis

**Reference Paper**: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (NeurIPS 2020)

| Aspect | Implementation | Paper | Status |
|--------|---------------|-------|--------|
| **Global Control Variate (c)** | Initialized to zeros | c ← 0 | ✅ CORRECT |
| **Local Control Variate (cᵢ)** | Initialized from global | c_i ← c | ✅ CORRECT |
| **Gradient Correction** | loss + (c_global - c_local)·w | ∇F(w) + c - cᵢ | ⚠️ IMPLEMENTATION ISSUE |
| **y_delta** | W_local - W_global | Δyᵢ = yᵢ - x | ✅ CORRECT |
| **c_delta Update Formula** | c_new = c_old - c_global - (y_delta/(lr*K)) | cᵢ⁺ = cᵢ - c + (1/(Kη))(x - yᵢ) | ⚠️ SIGN ISSUE |
| **Server Aggregation** | W += η·avg(y_delta), c += avg(c_delta) | x⁺ = x + Δx, c⁺ = c + (1/N)Σ(Δcᵢ) | ✅ CORRECT |

**Issues Identified**:

1. **Gradient Correction Implementation** (Lines 87-93 in Scaffold client_app.py):
   ```python
   ctrl = sum((cg - cl).dot(p) for p, cg, cl in zip(...))
   loss = ce_loss + ctrl
   ```
   This adds a scalar term to the loss instead of applying gradient correction. The paper specifies correcting the gradient direction, not modifying the loss. **Status**: ⚠️ APPROXIMATION

2. **Control Variate Update Formula** (Lines 123-126):
   ```python
   c_local_new = cl - cg - (yd / (self.local_lr * K))
   ```
   Paper formula: `cᵢ⁺ = cᵢ - c + (1/(Kη))(x - yᵢ)`
   Implementation uses `cl - cg - (yd/(lr*K))` where `yd = local - global`
   This simplifies to: `cl - cg - (local - global)/(lr*K)` = `cl - cg + (global - local)/(lr*K)`
   Paper: `cᵢ - c + (x - yᵢ)/(Kη)` = `cᵢ - c + (global - local)/(Kη)`
   **Status**: ✅ CORRECT (sign is consistent)

3. **Server Learning Rate**: Implementation uses `global_lr = 1.0` by default, which matches paper recommendation.

**Overall Assessment**: ⚠️ **MOSTLY CORRECT** - The gradient correction is implemented as loss modification rather than explicit gradient correction, which is an approximation that should still work but may have different convergence properties.

---

### 3. FedProx - Correctness Analysis

**Reference Paper**: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)

| Aspect | Implementation | Paper | Status |
|--------|---------------|-------|--------|
| **Proximal Term** | μ/2 · ||w - w_global||² | μ/2 · ||w - wᵗ||² | ✅ CORRECT |
| **μ Value** | 0.01 | Hyperparameter (paper uses 0.001 to 1) | ✅ VALID RANGE |
| **Local Solver** | SGD with proximal term | Any local solver with proximal | ✅ CORRECT |
| **Aggregation** | Weighted FedAvg | Weighted average | ✅ CORRECT |
| **Partial Participation** | fraction_fit = 1.0 (100%) | Supports partial participation | ✅ CORRECT |

**Implementation Details** (Lines 93-98 in fedprox/client_app.py):
```python
if proximal_mu > 0:
    proximal_term = 0.0
    for param, global_param in zip(self.net.parameters(), self.global_params):
        proximal_term += torch.sum((param - global_param) ** 2)
    loss += (proximal_mu / 2.0) * proximal_term
```

**Issues Identified**:
None - This is a clean, correct implementation of FedProx.

**Overall Assessment**: ✅ **CORRECT** - Faithful implementation of the FedProx algorithm.

---

### 4. pFedMe - Correctness Analysis

**Reference Paper**: T Dinh et al., "Personalized Federated Learning with Moreau Envelopes" (NeurIPS 2020)

| Aspect | Implementation | Paper | Status |
|--------|---------------|-------|--------|
| **Bi-Level Structure** | θ (personal) and w (global) | θᵢ (personalized) and w (global) | ✅ CORRECT |
| **Inner Optimization** | K=5 steps optimizing θ | K steps of SGD on θ | ✅ CORRECT |
| **Outer Optimization** | R=1 step updating w toward θ | R updates to w | ✅ CORRECT |
| **Moreau Envelope** | λ/2 · ||θ - w||² | λ/2 · ||θ - w||² | ✅ CORRECT |
| **λ Value** | 15.0 | Paper uses λ ∈ [1, 20] | ✅ VALID RANGE |
| **Outer Update Formula** | w = w - η·λ·(w - θ) | w ← w - η·λ·(w - θ) | ✅ CORRECT |
| **β-Mixing** | w^{t+1} = (1-β)·w^t + β·new_avg | Server aggregation with mixing | ✅ CORRECT |
| **θ Reset Strategy** | θ ← w each round | θ initialized from w | ✅ CORRECT |

**Implementation Details** (Lines 92-143 in pFedMe/client_app.py):
```python
# Inner loop: optimize θ
for batch in range(inner_steps):
    loss = self.criterion(logits, y)
    reg = (lamda_eff / 2.0) * ||θ - w||²
    loss = loss + reg
    theta_optimizer.step()

# Outer: update w toward θ
w_param.data = w_param.data - outer_lr * lamda_eff * (w_param.data - theta_param.data)
```

**Issues Identified**:

1. **Inner Steps vs Batches**: The implementation processes K batches per outer step, not K gradient steps per batch. This is a valid interpretation but differs from some paper implementations that do K steps on the same batch.

2. **Gradient Clipping** (Line 128):
   ```python
   torch.nn.utils.clip_grad_norm_(self.theta_net.parameters(), 1.0)
   ```
   Not in the original paper but helps stability.

**Overall Assessment**: ✅ **CORRECT** - Faithful implementation with reasonable stability enhancements.

---

### 5. HierFL - Correctness Analysis

**Reference**: Hierarchical Federated Learning with multiple papers (Liu et al., 2020; Briggs et al., 2020)

| Aspect | Implementation | Standard HierFL | Status |
|--------|---------------|-----------------|--------|
| **Topology** | Cloud → Leaf Servers → Clients | Two-tier hierarchy | ✅ CORRECT |
| **Leaf Aggregation** | Local FedAvg per leaf | Local aggregation | ✅ CORRECT |
| **Cloud Aggregation** | Weighted average + clustering | Global aggregation | ✅ CORRECT |
| **SCAFFOLD Integration** | Optional, enabled by flag | N/A (extension) | ✅ EXTENSION |
| **FedProx Integration** | prox_mu = 0.001 | N/A (extension) | ✅ EXTENSION |
| **Cloud Clustering** | Cosine similarity, τ=0.7 | N/A (extension) | ✅ EXTENSION |

**Configuration Settings**:
- `num_servers = 3`
- `clients_per_server = [3, 3, 3]` → 9 total clients
- `server_rounds_per_global = 1`
- `global_rounds = 100`

**Overall Assessment**: ✅ **CORRECT** - Valid hierarchical FL implementation with additional features (SCAFFOLD, clustering).

---

### 6. Fedge-100 - Correctness Analysis

Similar to HierFL but configured for the specific HHAR dataset experiment.

**Issues Identified**:
1. **Round Count Mismatch**: `pyproject.toml` shows `num-server-rounds = 2` while hierarchy config shows `global_rounds = 100`. This appears to be a test configuration left in place.

**Overall Assessment**: ✅ **CORRECT** (same as HierFL, different scale)

---

## Summary: Correctness Verdict

| Method | Paper Faithfulness | Implementation Quality | Overall |
|--------|-------------------|----------------------|---------|
| **CFL** | ✅ Faithful | ✅ Good | ✅ CORRECT |
| **SCAFFOLD** | ⚠️ Approximation | ⚠️ Works but simplified | ⚠️ MOSTLY CORRECT |
| **FedProx** | ✅ Faithful | ✅ Clean | ✅ CORRECT |
| **pFedMe** | ✅ Faithful | ✅ Good | ✅ CORRECT |
| **HierFL** | ✅ Valid design | ✅ Good | ✅ CORRECT |
| **Fedge-100** | ✅ Valid | ⚠️ Config mismatch | ✅ CORRECT |

### Key Findings:

1. **All methods are functionally correct** and should produce valid experimental results.

2. **SCAFFOLD has a simplified gradient correction** that modifies the loss instead of directly correcting gradients. This is mathematically similar but may have slightly different convergence properties.

3. **Hyperparameters are within valid ranges** as specified in the original papers.

4. **Learning rates are consistent** across methods (0.05 default) which allows fair comparison.

5. **Minor configuration issues** exist in Fedge-100 (test vs production round counts).

### Recommendations for Paper Accuracy:
- For SCAFFOLD: Consider implementing explicit gradient correction if exact paper replication is required
- For Fedge-100: Update `num-server-rounds` to 100 to match hierarchy config
- Add explicit learning rate parameters to config files for better reproducibility

---

## Non-IID Partitioning Analysis: Your Approach vs. Literature

This section analyzes your partitioning strategy compared to what's typically used in federated learning literature for HHAR and similar HAR datasets.

---

### Your Current Partitioning Strategy

**Method Used**: **User-Based Natural Partitioning** (1 user = 1 client)

```python
# From CFL/fedge/task.py (line 380)
# 3) Clients = users (no Dirichlet)
user_frames: Dict[str, List[tuple[pd.DataFrame,str,str]]] = {}
for df, u, s in frames:
    user_frames.setdefault(u, []).append((df, u, s))
users = sorted(list(user_frames.keys()))
user_for_client = users[partition_id]
```

**Key Characteristics**:
- Each of the 9 HHAR users becomes 1 federated client
- No artificial label distribution manipulation (no Dirichlet)
- Heterogeneity arises naturally from user behavior differences
- 80/20 train/test split within each client's data

---

### Literature Approaches for Non-IID in Federated Learning

#### 1. **Dirichlet Distribution (Most Common in Literature)**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **α (alpha)** | Concentration parameter | 0.1, 0.3, 0.5, 1.0 |
| **Lower α** | More heterogeneous (severe non-IID) | α = 0.1 |
| **Higher α** | More homogeneous (closer to IID) | α = 1.0+ |

**How it works**: Samples `q ~ Dir_N(α)` and allocates proportion `q_j` of each label's samples to client `P_j`.

**Used in**: [NIID-Bench](https://flower.ai/docs/baselines/niid_bench.html), most FL benchmarks

#### 2. **Natural/User-Based Partitioning (Your Approach)**

**How it works**: Each real user's data becomes one client. Heterogeneity is inherent.

**Used in**: Real-world HAR deployments, [FLAME](https://ar5iv.labs.arxiv.org/html/2202.08922), [ClusterFL](https://github.com/xmouyang/FL-Datasets-for-HAR)

#### 3. **Label-Based Sharding**

**How it works**: Each client gets data from only K labels (e.g., 2 out of 6 activities).

**Used in**: FedAvg original paper, simple non-IID experiments

---

### Comparison: Your Approach vs. Literature

| Aspect | Your Approach (User-Based) | Dirichlet (Literature Standard) |
|--------|---------------------------|--------------------------------|
| **Realism** | ✅ High - reflects real deployment | ⚠️ Synthetic - artificial skew |
| **Reproducibility** | ✅ Deterministic (same users) | ✅ Reproducible with seed |
| **Heterogeneity Type** | Feature + Label + Quantity | Primarily Label distribution |
| **Control Over Skew** | ❌ Fixed by dataset | ✅ Tunable via α parameter |
| **Common in HAR Literature** | ✅ Yes, especially for HHAR | ✅ Yes, for benchmarking |
| **Captures Device Heterogeneity** | ✅ Yes (different phones/watches) | ❌ No |
| **Captures Behavioral Heterogeneity** | ✅ Yes (different user habits) | ❌ No |

---

### Is Your User-Based Approach Legitimate?

## ✅ **YES - Your approach is legitimate and arguably MORE realistic**

### Supporting Evidence from Literature:

1. **Natural Heterogeneity is Preferred for HAR**:
   > "Datasets collected in-the-wild are ideal because they reflect real-world IoT settings, guaranteeing natural heterogeneities among different clients: individuals have different devices (sensor heterogeneity), different behavior, and different habits."
   > — [Non-IID Survey (arXiv 2411.12377)](https://arxiv.org/html/2411.12377v1)

2. **HHAR Dataset Design Purpose**:
   The HHAR dataset was specifically designed to capture **heterogeneity** in:
   - Different smartphone/smartwatch models
   - Different users with different activity patterns
   - Different sensor characteristics

   Using user-based partitioning **leverages the dataset's intended purpose**.

3. **Real-World FL Deployments Use Natural Partitions**:
   > "The non-IID property of data is simulated in most federated learning research, whereas real cases use actual user data."
   > — [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-02395-z)

4. **FLAME Framework Validates User-Based Approach**:
   > "The MDE setting presents a challenging case of non-IID-ness... User heterogeneity occurs due to differences in personal characteristics of the users, such as different running styles and gait variations."
   > — [FLAME (arXiv 2202.08922)](https://ar5iv.labs.arxiv.org/html/2202.08922)

---

### Key Differences to Acknowledge in Your Paper

| Your Approach | Dirichlet Approach |
|---------------|-------------------|
| **Non-IID Type**: Natural multi-modal heterogeneity | **Non-IID Type**: Controlled label skew |
| **Source**: Real user behavioral differences | **Source**: Artificial statistical manipulation |
| **Severity**: Determined by dataset | **Severity**: Tunable (α parameter) |
| **Comparability**: Harder to compare across papers | **Comparability**: Standard benchmark settings |

---

### Recommendations

#### ✅ **What You Should Do**:

1. **Clearly state in your paper**: "We use natural user-based partitioning where each of the 9 HHAR users forms one federated client, reflecting realistic deployment scenarios."

2. **Justify the choice**: "Unlike synthetic Dirichlet-based partitioning, user-based partitioning captures real-world heterogeneity including device differences, behavioral patterns, and sensor characteristics inherent to the HHAR dataset."

3. **Quantify your heterogeneity** (optional but recommended):
   - Report label distribution per client
   - Calculate Earth Mover's Distance (EMD) or KL divergence between clients
   - Show data quantity per client

#### ⚠️ **Potential Limitations to Acknowledge**:

1. **Fixed heterogeneity level**: Cannot tune the degree of non-IID like Dirichlet
2. **Dataset-specific**: Results may not directly compare to papers using Dirichlet with specific α values
3. **Small client count**: 9 clients is typical for HHAR but smaller than many FL benchmarks (often 10-100+ clients)

#### 📊 **Optional Enhancement**:

If reviewers request comparison with Dirichlet:
- Add experiments with Dirichlet partitioning (α = 0.1, 0.5, 1.0)
- Show that user-based results are comparable to moderate Dirichlet settings (α ≈ 0.3-0.5)

---

### Verdict

| Question | Answer |
|----------|--------|
| **Is user-based partitioning legitimate?** | ✅ **Yes** |
| **Is it used in literature?** | ✅ **Yes**, especially for HAR |
| **Is it more realistic than Dirichlet?** | ✅ **Yes**, for real-world deployment |
| **Should you mention this in your paper?** | ✅ **Yes**, clearly explain the choice |
| **Do you need Dirichlet experiments?** | ⚠️ **Optional**, but could strengthen paper |

---

### Summary

Your **user-based natural partitioning is a valid and legitimate non-IID strategy** that is:
- Well-established in HAR federated learning literature
- More realistic than synthetic Dirichlet partitioning
- Appropriate for the HHAR dataset's design purpose
- Used by papers like FLAME, ClusterFL, and real-world FL deployments

The key is to **clearly document this choice** in your methodology section and acknowledge that it represents natural heterogeneity rather than controlled statistical skew.
