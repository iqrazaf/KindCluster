#!/usr/bin/env python3
"""
baseline_comparison.py

Compare four placement strategies:

A) Optuna (full causal) – objective uses PN/PS/PNS + STIG + placement-sensitive latency
B) Optuna (no causal)  – objective uses latency-only (no PN/PS/PNS/STIG)
C) Genetic Algorithm baseline – latency-only objective
D) Reinforcement Learning baseline – latency-only objective

All four are *evaluated* using the same full-causal metric:
  - interference score from PN/PS/PNS + STIG
  - placement-sensitive latency
"""

import random
import re
from pathlib import Path
from typing import Dict, Tuple, Iterable, Set, List

import numpy as np
import pandas as pd
import optuna

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PNS_CSV = Path("data/Probability_Table/cross_shop_PNS_T3.csv")   # PN/PS/PNS table
RT_CSV  = Path("data/RT_data/RT_T3_S1.csv")                      # p95 latencies
STIK_DIR = Path("data/stigs_graphs")                               # directory of STIG CSVs

NODE_NAMES = ["kind-cluster-worker", "kind-cluster-worker2"]
NODE_COUNT = len(NODE_NAMES)

# latency factors
COLOC_FACTOR   = 0.8    # edge is cheaper if services colocated
CROSS_FACTOR   = 1.2    # edge is more expensive if across nodes
INTERF_LATENCY_SCALE = 1.0 / 100.0  # how much (PN+PS-PNS)*STIG inflates edge latency

NET_PENALTY_MS = 0.0    # we now model network via factors above; can keep 0
MIN_STRENGTH   = 0.0
REQUIRE_CROSS_APP = True

# GA hyper-params
GA_POP_SIZE  = 200
GA_GENS      = 250
GA_MUT_RATE  = 0.08
GA_CROSS_RATE = 0.8
GA_ELITE      = 0.08

# RL hyper-params
PG_ITERS         = 2000
PG_BATCH         = 256
PG_LR            = 0.5
PG_ENTROPY_COEF  = 0.01

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def trial_label_from_filename(p: Path, default: str = "") -> str:
    stem = p.stem
    m = re.search(r'(?:^|_)t(\w+)$', stem, flags=re.IGNORECASE)
    if m:
        return f"T{m.group(1)}"
    parts = stem.split("_")
    last = parts[-1]
    if re.fullmatch(r"[Tt]\w+", last):
        return last.upper()
    return default

def pretty(key: Tuple[str, str]) -> str:
    ns, short = key
    return f"{short}.{ns}"

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_latencies(csv_path: Path) -> Dict[Tuple[str, str], float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing RT file: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    for col in ("namespace", "service"):
        if col not in df.columns:
            raise ValueError("RT csv must contain 'namespace' and 'service' columns.")

    # choose p95 column
    p95_col = None
    for cand in ("p95_ms", "p95_seconds", "p95"):
        if cand in df.columns:
            p95_col = cand
            break
    if p95_col is None:
        raise ValueError("RT csv must contain one of: p95_ms, p95_seconds, p95")

    vals = pd.to_numeric(df[p95_col], errors="coerce")
    if p95_col == "p95_seconds":
        vals = vals * 1000.0
    else:
        # if small, assume seconds
        if vals.median(skipna=True) < 100:
            vals = vals * 1000.0
    df["p95_ms"] = vals

    df["namespace"] = df["namespace"].astype(str).str.strip().str.lower()
    df["service"]   = df["service"].astype(str).str.strip()
    df["short"]     = df["service"].apply(lambda s: str(s).split(".")[0].strip().lower())

    df = df.dropna(subset=["p95_ms"])
    df["key"] = list(zip(df["namespace"], df["short"]))
    med = df.groupby("key")["p95_ms"].median().to_dict()
    return {k: float(v) for k, v in med.items()}

def load_pn_ps_pns(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PNS file: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("PNS csv must contain X,Y columns.")

    has_app_cols = {"app(x)", "app(y)"}.issubset(df.columns)
    if REQUIRE_CROSS_APP and has_app_cols:
        df = df[df["app(x)"].str.lower() != df["app(y)"].str.lower()]

    for col in ("pn", "ps", "pns"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    if "pns" in df.columns and MIN_STRENGTH > 0:
        df = df[df["pns"] >= float(MIN_STRENGTH)]

    if has_app_cols:
        df["src_ns"] = df["app(x)"].astype(str).str.strip().str.lower()
        df["tgt_ns"] = df["app(y)"].astype(str).str.strip().str.lower()
    else:
        df["src_ns"] = ""
        df["tgt_ns"] = ""

    df["src_short"] = df["x"].astype(str).str.strip().str.lower()
    df["tgt_short"] = df["y"].astype(str).str.strip().str.lower()

    df["src_key"] = list(zip(df["src_ns"], df["src_short"]))
    df["tgt_key"] = list(zip(df["tgt_ns"], df["tgt_short"]))
    return df[["src_key", "tgt_key", "pn", "ps", "pns"]].copy()

def load_stig_weights(stik_dir: Path) -> Dict[Tuple[Tuple[str,str], Tuple[str,str]], float]:
    weights: Dict[Tuple[Tuple[str,str], Tuple[str,str]], float] = {}
    if not stik_dir.exists():
        print(f"[warn] STIK dir {stik_dir} missing; using weight 1.0 for all pairs.")
        return weights

    for csv_path in stik_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df = _normalize_columns(df)
        if not {"x", "y"}.issubset(df.columns):
            continue
        if "weight" not in df.columns and "stik" not in df.columns:
            continue
        wcol = "weight" if "weight" in df.columns else "stik"
        df[wcol] = pd.to_numeric(df[wcol], errors="coerce").fillna(1.0)
        for _, r in df.iterrows():
            sx = ("", str(r["x"]).strip().lower())
            sy = ("", str(r["y"]).strip().lower())
            w  = float(r[wcol])
            weights[(sx, sy)] = w
    if not weights:
        print("[warn] No valid STIG weight rows found; weight=1.0 will be used.")
    return weights

# -------------------------------------------------------------------
# Build problem
# -------------------------------------------------------------------
def build_problem():
    latency_ms = load_latencies(RT_CSV)
    pns_df = load_pn_ps_pns(PNS_CSV)
    stik_weights = load_stig_weights(STIK_DIR)

    # collect all services
    services_from_pns = set(pns_df["src_key"]).union(set(pns_df["tgt_key"]))
    all_services = sorted(set(latency_ms.keys()).union(services_from_pns))

    # fill missing latencies
    for k in all_services:
        if k not in latency_ms:
            latency_ms[k] = 1000.0

    # build pair list with STIK
    pairs = []
    for _, r in pns_df.iterrows():
        src = tuple(r["src_key"])
        tgt = tuple(r["tgt_key"])
        pn  = float(r["pn"])
        ps  = float(r["ps"])
        pns = float(r["pns"])
        # STIK is keyed only by short names; fall back to 1.0
        src_short = ("", src[1])
        tgt_short = ("", tgt[1])
        stik = stik_weights.get((src_short, tgt_short), 1.0)
        pairs.append({
            "src": src,
            "tgt": tgt,
            "pn": pn,
            "ps": ps,
            "pns": pns,
            "stik": stik,
        })

    comm_edges = sorted({tuple(sorted((p["src"], p["tgt"]))) for p in pairs if p["src"] != p["tgt"]})

    print(f"[diag] RT services: {len(latency_ms)} | pairs rows (PN/PS/PNS): {len(pairs)}")
    if pairs:
        ex = pairs[0]
        print("Example pair:",
              pretty(ex["src"]), "->", pretty(ex["tgt"]),
              f"PN={ex['pn']:.2f}, PS={ex['ps']:.2f}, PNS={ex['pns']:.2f}, STIK={ex['stik']:.3f}")

    return all_services, latency_ms, pairs, comm_edges

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def compute_metrics_full(mapping: Dict[Tuple[str,str], int],
                         latency_ms: Dict[Tuple[str,str], float],
                         pairs: List[Dict],
                         comm_edges: List[Tuple[Tuple[str,str],Tuple[str,str]]]) -> Tuple[float, float]:
    """
    Full-causal evaluation:
      - interference uses (PN+PS-PNS)*STIG, only when colocated
      - latency uses edge base latency scaled by coloc/cross factors and causal weight
    """
    total_interf = 0.0
    total_latency = 0.0
    default_ms = 1000.0

    for p in pairs:
        src = p["src"]; tgt = p["tgt"]
        if src not in mapping or tgt not in mapping:
            continue
        same = (mapping[src] == mapping[tgt])
        base = 0.5 * (latency_ms.get(src, default_ms) + latency_ms.get(tgt, default_ms))

        # causal piece
        expected = (p["pn"] + p["ps"] - p["pns"]) * p["stik"]
        expected = max(expected, 0.0)  # clamp; only positive makes interference
        if same:
            total_interf += expected

        factor = COLOC_FACTOR if same else CROSS_FACTOR
        # latency inflation from interference
        lat_weight = 1.0 + expected * INTERF_LATENCY_SCALE
        total_latency += base * factor * lat_weight

    # small network penalty for any comm edge crossing nodes (optional)
    for a, b in comm_edges:
        if a in mapping and b in mapping and mapping[a] != mapping[b]:
            total_latency += NET_PENALTY_MS

    return total_interf, total_latency

def compute_latency_only(mapping: Dict[Tuple[str,str], int],
                         latency_ms: Dict[Tuple[str,str], float],
                         pairs: List[Dict],
                         comm_edges: List[Tuple[Tuple[str,str],Tuple[str,str]]]) -> float:
    """
    Latency-only model (used as optimization objective for non-causal variants).
    Ignores PN/PS/PNS and STIG.
    """
    total_latency = 0.0
    default_ms = 1000.0

    for p in pairs:
        src = p["src"]; tgt = p["tgt"]
        if src not in mapping or tgt not in mapping:
            continue
        same = (mapping[src] == mapping[tgt])
        base = 0.5 * (latency_ms.get(src, default_ms) + latency_ms.get(tgt, default_ms))
        factor = COLOC_FACTOR if same else CROSS_FACTOR
        total_latency += base * factor

    for a, b in comm_edges:
        if a in mapping and b in mapping and mapping[a] != mapping[b]:
            total_latency += NET_PENALTY_MS

    return total_latency

# -------------------------------------------------------------------
# Optuna models
# -------------------------------------------------------------------
def run_optuna_full_causal(services, latency_ms, pairs, comm_edges):
    print("\n=== Running Model A: Optuna full causal ===")
    def objective(trial: optuna.Trial):
        mapping = {svc: trial.suggest_int(pretty(svc), 0, NODE_COUNT-1)
                   for svc in services}
        inter, lat = compute_metrics_full(mapping, latency_ms, pairs, comm_edges)
        return inter, lat

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["minimize","minimize"])
    study.optimize(objective, n_trials=200)

    best = study.best_trials[0]
    mapping = {svc: best.params[pretty(svc)] for svc in services}
    inter, lat = compute_metrics_full(mapping, latency_ms, pairs, comm_edges)

    print(f"Optuna full-causal best -> interference={inter:.2f}, latency={lat:.2f} ms")
    return mapping, (inter, lat)

def run_optuna_no_causal(services, latency_ms, pairs, comm_edges):
    print("\n=== Running Model B: Optuna no-causal (latency-only) ===")
    def objective(trial: optuna.Trial):
        mapping = {svc: trial.suggest_int(pretty(svc), 0, NODE_COUNT-1)
                   for svc in services}
        lat = compute_latency_only(mapping, latency_ms, pairs, comm_edges)
        return lat

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    best = study.best_trial
    mapping = {svc: best.params[pretty(svc)] for svc in services}
    # evaluate with full causal metric
    inter, lat = compute_metrics_full(mapping, latency_ms, pairs, comm_edges)
    print(f"Optuna no-causal eval -> interference={inter:.2f}, latency={lat:.2f} ms")
    return mapping, (inter, lat)

# -------------------------------------------------------------------
# GA baseline
# -------------------------------------------------------------------
def run_ga_baseline(services, latency_ms, pairs, comm_edges):
    print("\n=== Running Model C: GA baseline (latency-only) ===")
    n = len(services)
    idx_to_svc = {i: s for i, s in enumerate(services)}

    def random_chrom():
        return np.random.randint(0, NODE_COUNT, size=n, dtype=np.int32)

    def chrom_to_mapping(ch):
        return {idx_to_svc[i]: int(ch[i]) for i in range(n)}

    def fitness(ch):
        m = chrom_to_mapping(ch)
        return compute_latency_only(m, latency_ms, pairs, comm_edges)

    pop = [random_chrom() for _ in range(GA_POP_SIZE)]
    fitnesses = np.array([fitness(c) for c in pop], dtype=float)

    best_ch = None
    best_fit = float("inf")

    for g in range(GA_GENS):
        elite_n = max(1, int(GA_ELITE * GA_POP_SIZE))
        order = np.argsort(fitnesses)
        new_pop = [pop[i].copy() for i in order[:elite_n]]

        while len(new_pop) < GA_POP_SIZE:
            a, b = np.random.choice(GA_POP_SIZE, size=2, replace=False)
            p1 = pop[a] if fitnesses[a] < fitnesses[b] else pop[b]
            child = p1.copy()
            if np.random.rand() < GA_CROSS_RATE:
                p2 = pop[np.random.randint(0, GA_POP_SIZE)]
                mask = np.random.rand(n) < 0.5
                child[mask] = p2[mask]
            mut_mask = np.random.rand(n) < GA_MUT_RATE
            if mut_mask.any():
                child[mut_mask] = np.random.randint(0, NODE_COUNT, size=mut_mask.sum())
            new_pop.append(child)

        pop = new_pop
        fitnesses = np.array([fitness(c) for c in pop], dtype=float)
        idx = int(np.argmin(fitnesses))
        if fitnesses[idx] < best_fit:
            best_fit = float(fitnesses[idx])
            best_ch = pop[idx].copy()
        if (g+1) % max(1, GA_GENS//10) == 0:
            print(f"  GA gen {g+1}/{GA_GENS} best latency-only={best_fit:.2f}")

    mapping = chrom_to_mapping(best_ch)
    inter, lat = compute_metrics_full(mapping, latency_ms, pairs, comm_edges)
    print(f"GA baseline eval -> interference={inter:.2f}, latency={lat:.2f} ms")
    return mapping, (inter, lat)

# -------------------------------------------------------------------
# RL baseline (simple REINFORCE)
# -------------------------------------------------------------------
def run_rl_baseline(services, latency_ms, pairs, comm_edges):
    print("\n=== Running Model D: RL baseline (latency-only) ===")
    n = len(services)
    logits = np.zeros((n, NODE_COUNT), dtype=float)
    best_mapping = None
    best_lat = float("inf")
    baseline = 0.0

    def sample_one():
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        choices = [np.random.choice(NODE_COUNT, p=probs[i]) for i in range(n)]
        mapping = {services[i]: int(choices[i]) for i in range(n)}
        return mapping, probs

    def reward_of_mapping(m):
        lat = compute_latency_only(m, latency_ms, pairs, comm_edges)
        return -lat, lat  # reward, metric

    for it in range(PG_ITERS):
        batch_rewards = []
        batch_mappings = []
        batch_probs = []

        for _ in range(PG_BATCH):
            m, probs = sample_one()
            r, lat = reward_of_mapping(m)
            batch_rewards.append(r)
            batch_mappings.append(m)
            batch_probs.append(probs)
            if lat < best_lat:
                best_lat = lat
                best_mapping = m.copy()

        batch_rewards = np.array(batch_rewards, dtype=float)
        batch_mean = batch_rewards.mean()
        baseline = 0.9 * baseline + 0.1 * batch_mean if it > 0 else batch_mean

        grad = np.zeros_like(logits)
        for i, m in enumerate(batch_mappings):
            adv = batch_rewards[i] - baseline
            probs = batch_probs[i]
            chosen = np.array([m[services[j]] for j in range(n)], dtype=int)
            for s_idx in range(n):
                one_hot = np.zeros(NODE_COUNT, dtype=float)
                one_hot[chosen[s_idx]] = 1.0
                grad[s_idx] += adv * (one_hot - probs[s_idx])

        grad /= max(1.0, len(batch_mappings))
        logits += PG_LR * grad

        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1).mean()
        logits += PG_LR * PG_ENTROPY_COEF * (1.0 - entropy)

        if (it+1) % max(1, PG_ITERS//10) == 0:
            print(f"  PG iter {it+1}/{PG_ITERS}, best latency-only={best_lat:.2f}")

    mapping = best_mapping
    inter, lat = compute_metrics_full(mapping, latency_ms, pairs, comm_edges)
    print(f"RL baseline eval -> interference={inter:.2f}, latency={lat:.2f} ms")
    return mapping, (inter, lat)

# -------------------------------------------------------------------
# Helpers for printing
# -------------------------------------------------------------------
def print_mapping(label: str,
                  mapping: Dict[Tuple[str,str], int],
                  score: Tuple[float,float]):
    print(f"\n=== {label} result ===")
    print(f"Interference: {score[0]:.2f} | Latency: {score[1]:.2f} ms")
    nodes = {n: [] for n in NODE_NAMES}
    for svc, idx in mapping.items():
        nodes[NODE_NAMES[idx]].append(svc)
    for node in NODE_NAMES:
        print(node + ":")
        for s in sorted(nodes[node]):
            print("  -", pretty(s))

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def main():
    services, latency_ms, pairs, comm_edges = build_problem()

    print("\n=== Nodes ===", NODE_NAMES)
    print("Targets:", len(services), "| Pairs:", len(pairs))
    print("\nMedian p95 latencies (ms):")
    for s in sorted(services):
        print(f"  {pretty(s):30s} {latency_ms[s]:10.2f}")

    # A) Optuna full causal
    opt_causal_map, opt_causal_score = run_optuna_full_causal(services, latency_ms, pairs, comm_edges)
    print_mapping("Optuna full causal", opt_causal_map, opt_causal_score)

    # B) Optuna no causal
    opt_nocausal_map, opt_nocausal_score = run_optuna_no_causal(services, latency_ms, pairs, comm_edges)
    print_mapping("Optuna no causal", opt_nocausal_map, opt_nocausal_score)

    # C) GA baseline
    ga_map, ga_score = run_ga_baseline(services, latency_ms, pairs, comm_edges)
    print_mapping("GA baseline", ga_map, ga_score)

    # D) RL baseline
    rl_map, rl_score = run_rl_baseline(services, latency_ms, pairs, comm_edges)
    print_mapping("RL baseline", rl_map, rl_score)

    # final comparison table
    print("\n=== FINAL COMPARISON TABLE ===")
    print(f"Optuna (full causal)       : interference={opt_causal_score[0]:.2f}, latency={opt_causal_score[1]:.2f} ms")
    print(f"Optuna (no causal, lat-only): interference={opt_nocausal_score[0]:.2f}, latency={opt_nocausal_score[1]:.2f} ms")
    print(f"GA baseline                : interference={ga_score[0]:.2f}, latency={ga_score[1]:.2f} ms")
    print(f"RL baseline                : interference={rl_score[0]:.2f}, latency={rl_score[1]:.2f} ms")

if __name__ == "__main__":
    main()


