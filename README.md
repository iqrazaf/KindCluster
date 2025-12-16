## ----------------(PART-1-CLuster Deployment) ---------------------------------
# KindCluster 
Following is a cleaned, end-to-end, reproducible guide that combines kind cluster setup and k6 load test into one workflow.

It assumes a Ubuntu-like VM where you have sudo. 

Set up a local Kubernetes cluster with:

1: kind + 1 control-plane + 2 workers

2: Istio (demo profile)

3: Deploy Sock Shop, TeaStore, and Bookinfo apps

4: A single Istio Gateway exposing /sock, /tea, /book via http://localhost

5: A k6 spike-load test (s1_spike.js) that hits those three apps.

## ----------------(PART-2-CLuster Optimization Problem) ---------------------------------
# Service Optimization 
It includes 3 parts: 
(A) Optimization Problem (Optimization.py)
(B) Ablation Study (Ablation_study.py)
(C) Baseline Comparison with Optuna Optimizer method (baseline_comparison.py)

Terminology note: STIGs vs STIK:

You will see both **STIGs** and **STIK** in the code and folder names (for example, `STIK_DIR`).  
They refer to the **same concept**: **Spatio-Temporal-Interference-Graphs (STIGs)**.  
“STIK” is just a naming variation in the implementation, not a different method.

## (A) Optimization Problem (Optimization.py)

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

-----------------------------------END (A)-----------------------------

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
