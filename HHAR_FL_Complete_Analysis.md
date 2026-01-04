# Federated Learning for Human Activity Recognition

*Papers Using HHAR Dataset - Experimental Settings, Results & Non-IID Analysis*

---

## 1. Overview

This document summarizes 5 research papers that have specifically used HHAR (Heterogeneity Human Activity Recognition) dataset for federated learning experiments. It includes experimental settings, accuracy results, and detailed non-IID data partitioning strategies.

---

## 2. Summary of 5 HHAR Papers

| Paper | Dataset(s) | Model | Rounds | Epochs | FL Acc | Central Acc |
|-------|------------|-------|--------|--------|--------|-------------|
| Sozinov 2018 | HHAR | DNN/Softmax | 1000 | 1-5 | 78-89% | 83-93% |
| Meta-HAR 2021 | HHAR, USC-HAD | CNN+MAML | - | 5 | 75-90% | >97% |
| FedDist 2022 | HHAR, RealWorld, SHL | CNN | 200 | 5 | 74% | 92% |
| FedAli 2024 | HHAR, RealWorld | HART | 600 | - | Best | - |
| Gudur 2020 | HHAR (4 activities) | Distillation | - | - | +11% | - |

---

## 3. Detailed Experimental Settings

### 3.1 Sozinov et al. (2018)

**Paper:** "Human Activity Recognition Using Federated Learning" - IEEE BDCloud

| Parameter | Value/Description |
|-----------|-------------------|
| Dataset | HHAR - 9 users, 6 activities |
| Preprocessing | Window: 3s with 50% overlap, resampled to 65Hz, 32 statistical features |
| Models | DNN: 2 hidden layers, 100 neurons each (24,006 params); Softmax: 198 params |
| Learning Rate | 0.001 |
| Batch Size | DNN: 100; Softmax: 50 |
| Local Epochs | DNN: 1; Softmax: 5 |
| Comm. Rounds | Up to 1000 |
| DNN Accuracy | Non-IID: 87% FL vs 93% centralized; Uniform: 89% FL vs 93% centralized |
| Softmax Accuracy | Non-IID: 78% FL vs 83% centralized; Uniform: 80% FL vs 83% centralized |

### 3.2 Meta-HAR (Li et al., 2021)

**Paper:** "Meta-HAR: Federated Representation Learning for Human Activity Recognition" - WWW 2021

| Parameter | Value/Description |
|-----------|-------------------|
| Datasets | HHAR, USC-HAD, newly collected dataset |
| Architecture | Signal embedding network (MAML) + personalized classification network |
| Optimizer | Adam |
| Local Epochs | 5 epochs per round |
| Loss Function | Pairwise loss (embedding) + Cross-entropy (classification) |
| Central Accuracy | >97% on HHAR |
| FedAvg Accuracy | <80% on HHAR (signal heterogeneity causes ~20% drop) |
| Meta-HAR Accuracy | 75.83% (HHAR meta-test), +24.7% over FedReptile |

### 3.3 FedDist (Ek et al., 2022)

**Paper:** "Evaluation and comparison of FL algorithms for HAR on smartphones" - PMC

| Parameter | Value/Description |
|-----------|-------------------|
| Datasets | HHAR (51 clients), RealWorld (15 clients), SHL (9 clients) |
| Model | CNN architecture |
| Optimizer | Mini-batch SGD |
| Batch Size | 32 |
| Dropout | 0.50 |
| Comm. Rounds | 200 |
| Local Epochs | 5 |
| Train/Test Split | 80%/20% per client |
| FedDist Accuracy | Generalization: 74.23%, Global F-Score: best among FL methods |

### 3.4 FedAli (Ek et al., 2024)

**Paper:** "FedAli: Personalized Federated Learning with Aligned Prototypes" - arXiv

| Parameter | Value/Description |
|-----------|-------------------|
| Datasets | HHAR (36 clients), RealWorld (105 clients), Combined (141 clients) |
| Model | HART transformer: 6 encoder blocks, projection size 192 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 64 |
| Comm. Rounds | 600 |
| Train/Test Split | 80%/20% class-stratified |
| FedAli Results | Best on HHAR; >20% generalization gain with pre-training |

### 3.5 Gudur et al. (2020)

**Paper:** "Federated Learning with Heterogeneous Labels and Models" - NeurIPS MLMH Workshop

| Parameter | Value/Description |
|-----------|-------------------|
| Dataset | HHAR with 4 activities (subset) |
| Hardware | Raspberry Pi 2 (on-device FL) |
| Method | Model Distillation Update with federated label-based aggregation |
| Innovation | Transfer model scores instead of weights; handles label heterogeneity |
| Accuracy Gain | ~11.01% average deterministic accuracy increase |

---

## 4. Non-IID Data Partitioning Strategies

This section details how each of the 5 HHAR papers implemented non-IID data partitioning. Understanding these strategies is crucial for reproducing results and designing new experiments.

### 4.1 Sozinov et al. (2018) - Three Non-IID Settings

#### Setting A: Non-IID & Unbalanced (Pathological)

| Aspect | Description |
|--------|-------------|
| Total Clients | 27 clients |
| Partitioning | Each of 9 users split into 3 clients (9 × 3 = 27) |
| Activities/Client | 2 activities only |
| Data Balance | One activity has 50% less data than the other |
| Example | Client 1a: [Walking 100%, Sitting 50%]; Client 1b: [Standing 100%, Biking 50%] |
| Non-IID Type | Label skew + Quantity imbalance |

#### Setting B: Uniform Distribution

| Aspect | Description |
|--------|-------------|
| Total Clients | 9 clients |
| Partitioning | Each user = 1 client |
| Activities/Client | 6 activities (all) |
| Data Balance | Equal samples per activity |
| Non-IID Type | Signal heterogeneity only (different users/devices) |

#### Setting C: Skewed Distribution

| Aspect | Description |
|--------|-------------|
| Total Clients | 9 clients |
| Partitioning | 1 skewed client + 8 uniform clients |
| Skewed Client | Only 1 activity (e.g., Walking only) |
| Other Clients | 6 activities each (uniform) |
| Non-IID Type | Extreme local specialization |

### 4.2 Meta-HAR (2021) - Activity Removal Strategy

| Aspect | Description |
|--------|-------------|
| Strategy | Remove 0-2 random activities per user |
| Filtering | Users with <3 activities removed from study |
| Result | 48 users remaining (for merged dataset) |
| Activities/Client | 4-6 activities (variable) |
| Non-IID Type | Missing labels (activity heterogeneity) |
| Key Finding | Signal heterogeneity (different users/sensors) causes ~20% accuracy drop: Centralized >97% vs FedAvg <80% |

### 4.3 FedDist (2022) - Device-Based Partitioning

| Aspect | Description |
|--------|-------------|
| Partitioning Logic | Each (user, device) combination = 1 client |
| HHAR Devices | 8 smartphones + 4 smartwatches per user |
| Total Clients | 51 clients for HHAR |
| Activities/Client | 6 activities (all) |
| Train/Test Split | 80%/20% per client |
| Non-IID Sources | Device heterogeneity + User behavior + Sampling rate (50-200Hz) |

### 4.4 FedAli (2024) - Consolidated Device Partitioning

| Aspect | Description |
|--------|-------------|
| Partitioning Logic | Each (user, device_type) = 1 client (identical models combined) |
| Device Types | 4 types: LG Nexus 4, Samsung S3, S3 mini, S+ (2 of each per user) |
| Total Clients | 36 clients for HHAR (9 users × 4 device types) |
| Activities/Client | 6 activities (all) |
| Combined Dataset | HHAR (36) + RealWorld (105) = 141 clients |
| Non-IID Sources | Device type + User + Position (RealWorld) + Activity set differences |

### 4.5 Gudur et al. (2020) - Label Heterogeneity

| Aspect | Description |
|--------|-------------|
| Activity Subset | 4 activities (subset of 6 for clearer analysis) |
| Key Innovation | Different clients may have DIFFERENT label sets |
| Example | Client 1: {Walk, Sit, Stand}; Client 2: {Sit, Stand, Bike} |
| Challenge | How to aggregate models with different output classes? |
| Solution | Model Distillation - transfer scores for overlapping activities |
| Non-IID Type | Label space heterogeneity (different labels per client) |

---

## 5. Non-IID Partitioning Summary

| Paper | Partitioning | Clients | Activities | Non-IID Type |
|-------|--------------|---------|------------|--------------|
| **Sozinov (Pathological)** | User split into 3 | 27 | 2 | Label skew + Quantity imbalance |
| **Sozinov (Uniform)** | User-based | 9 | 6 | Signal heterogeneity only |
| **Sozinov (Skewed)** | 1 extreme + 8 uniform | 9 | 1 or 6 | Extreme local specialization |
| **Meta-HAR** | Activity removal | 48 | 4-6 | Missing labels + Signal heterogeneity |
| **FedDist** | (User, Device) | 51 | 6 | Device + User heterogeneity |
| **FedAli** | (User, Device_type) | 36 | 6 | Device type + User heterogeneity |
| **Gudur** | Label heterogeneity | - | Variable | Different label spaces per client |

---

## 6. Comparison with Current Implementation

### 6.1 Your Current Non-IID Setup

| Parameter | Your Implementation |
|-----------|---------------------|
| Total Clients | 9 |
| Partitioning | User-based (each of 9 HHAR users = 1 client) |
| Activities/Client | 6 (all activities) |
| Data Balance | Natural distribution |
| Non-IID Source | Signal heterogeneity only (different users/devices) |
| HierFL Setup | 3 servers × 3 clients = 9 total |

**Closest Match: Sozinov's Uniform Setting** - This is the LEAST non-IID setting (only ~4-6% accuracy drop reported).

### 6.2 Gap Analysis: Your Settings vs HHAR Papers

| Parameter | Your Setup | Sozinov | FedDist | FedAli |
|-----------|------------|---------|---------|--------|
| Clients | 9 | 9-27 | 51 | 36 |
| Rounds | 100 | 1000 | 200 | 600 |
| Learning Rate | 0.05 | 0.001 | SGD | 1e-4 |
| Batch Size | 32 | 50-100 | 32 | 64 |
| Local Epochs | 5 | 1-5 | 5 | - |
| Model | 1D CNN (23K) | DNN (24K) | CNN | HART Transformer |

**Legend:** Green = Match, Yellow = Partial match, Red = Significant difference

---

## 7. Recommendations for Scaling

Based on the 5 HHAR papers, here are recommended non-IID configurations for scaling your experiments:

### 7.1 Option A: FedAli Style (RECOMMENDED)

| Parameter | Configuration |
|-----------|---------------|
| Total Clients | 36 clients |
| Partitioning | Each (user, device_type) = 1 client → 9 users × 4 device types |
| Activities/Client | 6 (all activities) |
| Server Option 1 | 3 servers: Users 1-3, Users 4-6, Users 7-9 (12 clients each) |
| Server Option 2 | 4 servers by device type: Nexus 4, S3, S3 mini, S+ (9 clients each) |
| Non-IID Type | Natural device + user heterogeneity |
| Published | Yes - FedAli 2024 |

### 7.2 Option B: Sozinov Pathological

| Parameter | Configuration |
|-----------|---------------|
| Total Clients | 27 clients |
| Partitioning | Split each user into 3 → 9 × 3 = 27 |
| Activities/Client | 2 activities (one with 50% less data) |
| Server Setup | 3 servers: 9 clients each (by user groups) |
| Non-IID Type | Label skew + Quantity imbalance (challenging) |
| Published | Yes - Sozinov 2018 |

### 7.3 Option C: FedDist Style

| Parameter | Configuration |
|-----------|---------------|
| Total Clients | 51 clients |
| Partitioning | Each (user, device) = 1 client (every device separate) |
| Activities/Client | 6 activities |
| Server Setup | 5+ servers needed (e.g., ~10 clients each) |
| Non-IID Type | Device + User heterogeneity |
| Published | Yes - FedDist 2022 |

### 7.4 Final Recommendation

**For your HierFL experiments with increased servers and clients, use FedAli Style (36 clients) because:**

- Published in peer-reviewed venue (FedAli 2024)
- Natural non-IID (no artificial manipulation)
- 36 clients divides evenly into 3, 4, 6, 9, or 12 servers
- All 6 activities preserved (fair comparison)
- Realistic device heterogeneity scenario

---

## 8. References

**[1]** Sozinov, K., Vlassov, V., & Girdzijauskas, S. (2018). Human Activity Recognition Using Federated Learning. IEEE BDCloud.

**[2]** Li, C., Niu, D., Jiang, B., Zuo, X., & Yang, J. (2021). Meta-HAR: Federated Representation Learning for Human Activity Recognition. WWW 2021.

**[3]** Ek, S., Portet, F., Lalanda, P., & Vega, G. (2022). Evaluation and comparison of FL algorithms for HAR on smartphones. PMC.

**[4]** Ek, S., Wang, K., Portet, F., Lalanda, P., & Cao, J. (2024). FedAli: Personalized Federated Learning with Aligned Prototypes. arXiv.

**[5]** Gudur, G.K., et al. (2020). Federated Learning with Heterogeneous Labels and Models. NeurIPS MLMH Workshop.
