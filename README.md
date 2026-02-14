# KindCluster Artifact: Interference-Resilient Microservices Placement (End-to-End Guide)

This repository provides a reproducible artifact for evaluating **interference-aware microservice placement** in a local Kubernetes cluster.

It includes:

1. **Cluster deployment**: Kind (1 control-plane + 2 workers) + Istio + 3 microservice applications  
2. **Load testing**: k6 spike-load to generate latency and monitoring signals  
3. **Optimization pipeline**: cross-application causal probabilities (PN/PS/PNS) + STIGs + Optuna (NSGA-II)  
4. **Ablation and baselines**: quantify contributions and competitiveness  
---

## To deploy a fresh cluster
This step is written for an **Ubuntu-like VM** with `sudo` access. VM is only for deploying a cluster and collecting data. 
### 1) Deploy cluster + apps

bash 
chmod +x run_kind_cluster.sh

./run_kind_cluster.sh

## What the kind_cluster.sh is doing:

1: kind + 1 control-plane + 2 workers

2: Istio (demo profile)

3: Deploy Sock Shop, TeaStore, and Bookinfo apps

4: A single Istio Gateway exposing /sock, /tea, /book via http://localhost

If you would like more details about commands, please follow Kind_Cluster_script.pdf.

#  Load test to collect monitoring data (already exists in the data folder for paper experiments)
What the load_test.sh is doing: A k6 spike-load test (s1_spike.js) that hits those three apps.
 
Run the following commands to generate the load on the cluster. 
 
chmod +x load_test.sh

./load_test.sh
## ----------------(PART-2-CLuster Optimization Problem) on any Python tool (not on VM)-----------------
Use the existing collected data in the data and stigs_graphs folder:
# Service Optimization 
It includes 3 parts: 
(A-1) Cross Application Analysis for Causal Probabilities (PS,PN,PNS) (cross_app_analysis.py)
(A-2)Optimization Problem (Optimization.py)
(B) Ablation Study (Ablation_study.py)
(C) Baseline Comparison with Optuna Optimizer method (baseline_comparison.py)

Terminology note: STIGs vs STIK:

You will see both **STIGs** and **STIK** in the code and folder names (for example, `STIK_DIR`).  
They refer to the **same concept**: **Spatio-Temporal-Interference-Graphs (STIGs)**. “STIK” is just a naming variation in the implementation, not a different method. 

If you want to generate new STIGs graphs, follow our previous work for STIGs generation (https://dl.acm.org/doi/abs/10.1145/3629527.3653664) for more details.  

## (A-1) Cross Application Analysis for Causal Probabilities (PS,PN,PNS)  (cross_app_analysis.py)
PN/PS/PNS Table Generator (Cross-App Causal Probabilities from Traces): 

This script builds a **cross-application PN/PS/PNS probability table** from cleaned distributed traces. The output CSV is used later by the placement optimizers to quantify **directional interference likelihood** between services across different apps (for example Bookinfo ↔ TeaStore ↔ Sock-Shop).

It reads span-level traces, detects anomalous spans, and then computes **cross-app causal probabilities** using temporal co-occurrence within a time window Δt.

---
What this script produces: 

**Output file**
- `Probability_Table/cross_shop_PNS_T3.csv`

Each row represents a directed pair **X → Y**, where anomalies in Y are temporally associated with anomalies in X.

Columns include:
- `X`, `Y`
- `App(X)`, `App(Y)`
- `P(Y|X)` and `P(Y|X')`
- `PNS`, `PN`, `PS`
- `X_events`, `Y_events`

These values are written as **percentages (0–100)** in the output.


-----------------------------------END (A-1)-----------------------------
## (A-2) Optimization Problem (Optimization.py)

#Interference-Aware Service Placement Optimizer (Optuna + STIGs): 

This step contains `Optimization.py`, a reproducible script that searches for a **microservice-to-node placement** that minimizes:

1) **Total interference** (causal + spatio-temporal), and  
2) **Total latency** (p95 response time + cross-node network penalty)

It uses **Optuna (NSGA-II)** to produce a **Pareto front** for the interference–latency trade-off.

---

#What the script does:

At a high level:
)
- Loads **p95 latency** per service (from `RT_T3_S1.csv`, Different test Trials, each csv RT values belongs to a single Trial(e.g T1,T2...T5))
- Loads **STIG graphs** (as `*.gpickle` NetworkX graphs) and aggregates edge weights (Pre-generated interference graphs)
- Loads a **PN/PS/PNS probability table** (from `cross_shop_PNS_T3.csv`, different test Trials, each csv PNS values belongs to a single Trial(e.g T1,T2...T5)))
- Joins PN/PS/PNS with STIG weights to compute an **expected interference score** per (source, target) pair:
  - `expected = (PS + PN - PNS) * STIG_weight`
- Runs multi-objective optimization to find placements that minimize:
  - Objective 1: sum of expected interference for colocated interfering pairs
  - Objective 2: sum of p95 latencies + network penalties for cross-node communication edges
- Prints the best selected placement and saves a Pareto plot as an image.

-----------------------------------END (A-2)-----------------------------

## (B) Ablation Study (Ablation_study.py)
Ablation Study: Causal vs Non-Causal Placement Optimization (Optuna + STIGs):

This section includes an ablation script that compares **four objective variants** for microservice placement across cluster nodes. The goal is to show how each causal component contributes to better interference-aware placements.

The script uses **Optuna (NSGA-II)** to optimize a **two-objective** problem in every variant:

1) **Total interference** (definition depends on the ablation mode)  
2) **Total latency** (placement-dependent, includes imbalance penalty + cross-node comm penalty)

---

#What this script demonstrates:

The script runs the same optimization loop under four modes:

- **full**: uses the full causal signal  
  - `expected = (PN + PS − PNS) × STIG_weight`
- **no_probs**: removes causal probabilities (PN/PS/PNS)  
  - interference uses **STIG weight only**
- **no_stig**: removes STIG weights  
  - interference uses **(PN + PS − PNS) only**
- **latency_only**: removes interference completely  
  - only latency objective remains (still multi-objective, but interference is always 0)

Each mode produces:
- a best placement (printed)
- a Pareto front plot
- a final summary line: best interference + best latency


-----------------------------------END (B)-----------------------------
## (C) Baseline Comparison with Optuna Optimizer method (baseline_comparison.py)
 Baseline Comparison: Optuna vs GA vs RL (All Evaluated with Full-Causal Metric):

This script (`baseline_comparison.py`) compares **four placement strategies** for microservices across cluster nodes. The point is not just “who optimizes fastest,” but **who finds placements that remain good under a consistent interference-aware evaluation**.

It runs:

A) **Optuna (full causal)**: optimizes interference + latency using PN/PS/PNS + STIG  
B) **Optuna (no causal)**: optimizes latency-only (ignores PN/PS/PNS and STIG)  
C) **Genetic Algorithm (GA) baseline**: optimizes latency-only  
D) **Reinforcement Learning (RL) baseline**: optimizes latency-only (simple REINFORCE)

**Important:** even though B/C/D optimize latency-only, **all four are evaluated using the same full-causal metric** at the end:
- Interference score: `(PN + PS − PNS) × STIG` (only when colocated)
- Placement-sensitive latency: edge latency adjusted by colocated/cross-node factors and inflated by interference

-----------------------------------END (C)-----------------------------
## Repository layout

Expected structure (you can rename folders, but then update the paths at the top of the script)
