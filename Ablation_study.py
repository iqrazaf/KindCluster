from __future__ import annotations
import random
import re
from pathlib import Path
from typing import Dict, Tuple, Iterable, List, Set
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paths and directories
# -------------------------

PNS_CSV = Path("data/Probability_Table/cross_shop_PNS_T3.csv")
RT_CSV = Path("data/RT_data/RT_T3_S1.csv")
STIK_DIR = Path("data/stigs_graphs")   # folder where STIGs .gpickle files are stored

# -------------------------
# Cluster nodes
# -------------------------

NODE_NAMES = ["kind-cluster-worker", "kind-cluster-worker2"]
NODE_COUNT = len(NODE_NAMES)

# -------------------------
# Network penalty (latency)
# -------------------------

NET_PENALTY_MS = 50.0
MIN_STRENGTH = 0.0
REQUIRE_CROSS_APP = True

# convexity of node-latency cost ( >1 to penalise unbalanced placements )
LATENCY_ALPHA = 1.3

# -------------------------
# Service name mapping (STIG → trace short names)
# -------------------------

SERVICE_NAME_MAP = {
    # bookinfo
    "product-page": "productpage",
    "details": "details",
    "reviews": "reviews",
    "ratings": "ratings",

    # teastore
    "web-ui": "teastore-webui",
    "presis": "teastore-persistence",
    "auth": "teastore-auth",
    "image": "teastore-registry",

    # sock-shop
    "front-end": "front-end",
    "orders": "orders",
    "ship": "shipping",
    "queue": "queue-master",
    "user": "user",
}


# =========================
#   Helper functions
# =========================

def trial_label_from_filename(p: Path, default: str = "") -> str:
    stem = p.stem
    m = re.search(r"(?:^|_)t(\w+)$", stem, flags=re.IGNORECASE)
    if m:
        return f"T{m.group(1)}"
    parts = stem.split("_")
    last = parts[-1]
    if re.fullmatch(r"[Tt]\w+", last):
        return last.upper()
    return default


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def key_to_str(k: Tuple[str, str]) -> str:
    """(app, short) → 'app::short' for Optuna parameter name."""
    return f"{k[0]}::{k[1]}"


def pretty(svc_key: Tuple[str, str]) -> str:
    """Human readable form: 'short.app'."""
    app, short = svc_key
    return f"{short}.{app}"


def map_service_name_from_stig(name: str) -> str:
    """
    Map node labels from STIG graphs to the short service names
    used in the probability table (e.g. 'product-page: S1' → 'productpage').
    """
    base = str(name).split(":")[0].strip()
    mapped = SERVICE_NAME_MAP.get(base, base)
    return mapped.lower()


# =========================
#   Load latencies (p95 ms)
# =========================

def load_latencies(csv_path: Path) -> Dict[Tuple[str, str], float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing RT file: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    for col in ("namespace", "service"):
        if col not in df.columns:
            raise ValueError("RT csv must contain 'namespace' and 'service' columns.")

    # p95 column
    p95_col = None
    for cand in ("p95_ms", "p95_seconds", "p95"):
        if cand in df.columns:
            p95_col = cand
            break
    if p95_col is None:
        raise ValueError("RT csv must contain one of: p95_ms, p95_seconds, or p95")

    df["namespace"] = df["namespace"].astype(str).str.strip().str.lower()
    df["service"] = df["service"].astype(str).str.strip()

    vals = pd.to_numeric(df[p95_col], errors="coerce")
    if p95_col == "p95_seconds":
        vals = vals * 1000.0
    else:
        # heuristic: if looks like seconds, convert to ms
        if vals.median(skipna=True) < 100:
            vals = vals * 1000.0
    df["p95_ms"] = vals

    df["short"] = df["service"].apply(lambda s: str(s).split(".")[0].strip().lower())
    df = df.dropna(subset=["p95_ms"])

    df["key_app_short"] = list(zip(df["namespace"], df["short"]))
    med = df.groupby("key_app_short")["p95_ms"].median().to_dict()

    return {(app, short): float(ms) for (app, short), ms in med.items()}


# =========================
#   Load STIG weights
# =========================

def load_stig_weights(stik_dir: Path) -> Dict[Tuple[str, str], float]:
    import networkx as nx

    if not stik_dir.exists():
        raise FileNotFoundError(f"STIK directory not found: {stik_dir}")

    weight_lists: Dict[Tuple[str, str], List[float]] = {}

    for gpath in stik_dir.glob("*.gpickle"):
        g = nx.read_gpickle(gpath)
        for u, v, data in g.edges(data=True):
            if data.get("type") != "interference":
                continue
            w = float(data.get("weight", 0.0))
            if w <= 0:
                continue
            src_short = map_service_name_from_stig(u)
            tgt_short = map_service_name_from_stig(v)
            key = (src_short, tgt_short)
            weight_lists.setdefault(key, []).append(w)

    agg: Dict[Tuple[str, str], float] = {}
    for key, ws in weight_lists.items():
        agg[key] = float(np.mean(ws))

    return agg


# =========================
#   Load PN/PS/PNS + STIG
# =========================

def load_interference_with_stig(
    csv_path: Path,
    stik_weights: Dict[Tuple[str, str], float]
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PNS file: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("PNS csv must contain 'X' and 'Y' columns.")

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
        df["src_app"] = df["app(x)"].astype(str).str.strip().str.lower()
        df["tgt_app"] = df["app(y)"].astype(str).str.strip().str.lower()
    else:
        df["src_app"] = ""
        df["tgt_app"] = ""

    df["x_short"] = df["x"].astype(str).str.strip().str.lower()
    df["y_short"] = df["y"].astype(str).str.strip().str.lower()

    rows = []
    mapped_pairs = 0

    for _, r in df.iterrows():
        sx = r["x_short"]
        sy = r["y_short"]
        stik_key = (sx, sy)
        if stik_key not in stik_weights:
            continue

        mapped_pairs += 1
        stik_w = stik_weights[stik_key]
        factor = float(r["ps"]) + float(r["pn"]) - float(r["pns"])
        expected = factor * stik_w  # full causal expected interference

        src_key = (r["src_app"], sx)
        tgt_key = (r["tgt_app"], sy)

        rows.append(
            {
                "src_key": src_key,
                "tgt_key": tgt_key,
                "pn": float(r["pn"]),
                "ps": float(r["ps"]),
                "pns": float(r["pns"]),
                "stik": stik_w,
                "expected": expected,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=["src_key", "tgt_key", "pn", "ps", "pns", "stik", "expected"],
    )

    print(f"[diag] pairs rows (with STIK): {len(out)} | mapped pairs: {mapped_pairs}")
    if not out.empty:
        print("Example mapped pairs (short names shown as X->Y):")
        for _, r in out.head(3).iterrows():
            src_app, src_short = r["src_key"]
            tgt_app, tgt_short = r["tgt_key"]
            print(
                f"  {src_short}.{src_app} -> {tgt_short}.{tgt_app} | "
                f"PN={r['pn']:.2f}, PS={r['ps']:.2f}, PNS={r['pns']:.2f}, STIK={r['stik']:.3f}"
            )

    return out


# =========================
#   Build problem
# =========================

def build_problem():
    # Latencies
    latency_ms = load_latencies(RT_CSV)
    services_with_rt = set(latency_ms.keys())

    # STIG
    stig_weights = load_stig_weights(STIK_DIR)

    # PN/PS/PNS joined with STIG
    pairs_df = load_interference_with_stig(PNS_CSV, stig_weights)

    involved: Set[Tuple[str, str]] = set(pairs_df["src_key"]).union(set(pairs_df["tgt_key"]))
    target_services = sorted(list(services_with_rt.union(involved)))

    # backfill missing RT
    for k in target_services:
        if k not in latency_ms:
            latency_ms[k] = 1000.0

    # interference dict: src -> tgt -> {"expected", "pn", "ps", "pns", "stig"}
    interference: Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, float]]] = {}
    for _, r in pairs_df.iterrows():
        s = tuple(r["src_key"])
        t = tuple(r["tgt_key"])
        interference.setdefault(s, {})[t] = {
            "expected": float(r["expected"]),
            "pn": float(r["pn"]),
            "ps": float(r["ps"]),
            "pns": float(r["pns"]),
            "stik": float(r["stik"]),
        }

    # communication edges (undirected) for latency penalty
    comm_edges: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    for _, r in pairs_df.iterrows():
        a = tuple(r["src_key"])
        b = tuple(r["tgt_key"])
        if a != b:
            comm_edges.add(tuple(sorted((a, b))))

    print(
        f"[diag] RT services: {len(services_with_rt)} | "
        f"pairs rows (STIK): {len(pairs_df)} | mapped pairs: {sum(len(v) for v in interference.values())}"
    )
    if interference:
        print("Example mapped pairs (short names shown as X->Y):")
        shown = 0
        for src, tgts in interference.items():
            for tgt, vals in tgts.items():
                print(
                    f"  {pretty(src)} -> {pretty(tgt)} | "
                    f"PN={vals['pn']:.2f}, PS={vals['ps']:.2f}, "
                    f"PNS={vals['pns']:.2f}, STIK={vals['stik']:.3f}"
                )
                shown += 1
                if shown >= 3:
                    break
            if shown >= 3:
                break

    return target_services, latency_ms, interference, sorted(list(comm_edges))


# =========================
#   Service priorities
# =========================

def compute_service_priorities(
    target_services: List[Tuple[str, str]],
    interference: Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, float]]],
) -> Dict[Tuple[str, str], float]:
    scores = {svc: 0.0 for svc in target_services}
    for src, tgts in interference.items():
        for tgt, vals in tgts.items():
            w = float(vals.get("expected", 0.0))
            scores[src] = scores.get(src, 0.0) + w
            scores[tgt] = scores.get(tgt, 0.0) + w
    return scores


# =========================
#   Latency computation
# =========================

def compute_latency(
    target_services: List[Tuple[str, str]],
    latency_ms: Dict[Tuple[str, str], float],
    placement_idx: Dict[Tuple[str, str], int],
    comm_edges: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    alpha: float = LATENCY_ALPHA,
) -> float:
    """
    Placement-dependent latency:
      1) For each node, sum p95_ms of services placed on it -> node load
      2) Penalise unbalanced placements via convex cost: load**alpha
      3) Add fixed penalty for cross-node communication edges
    """
    node_load = {i: 0.0 for i in range(NODE_COUNT)}
    for svc in target_services:
        node = placement_idx[svc]
        node_load[node] += latency_ms.get(svc, 1000.0)

    total_latency = 0.0
    for n in range(NODE_COUNT):
        load = node_load[n]
        if load > 0:
            total_latency += load ** alpha

    for a, b in comm_edges:
        if a in placement_idx and b in placement_idx and placement_idx[a] != placement_idx[b]:
            total_latency += NET_PENALTY_MS

    return total_latency


# =========================
#   Objectives (variants)
# =========================

def make_objective_variant(
    target_services,
    latency_ms,
    interference,
    comm_edges,
    mode: str,
):
    """
    mode:
      - "full"          : expected = (PN+PS−PNS) * STIG
      - "no_probs"      : expected = STIG only (drops PN/PS/PNS)
      - "no_stig"       : expected = PN+PS−PNS only (drops STIG)
      - "latency_only"  : ignores interference completely
    """
    def evaluate(trial: optuna.Trial):
        placement_idx: Dict[Tuple[str, str], int] = {}
        for svc in target_services:
            key_str = key_to_str(svc)
            placement_idx[svc] = trial.suggest_int(key_str, 0, NODE_COUNT - 1)

        total_interference = 0.0

        if mode != "latency_only":
            for src, tgts in interference.items():
                if src not in placement_idx:
                    continue
                for tgt, vals in tgts.items():
                    if tgt not in placement_idx:
                        continue
                    colocated = placement_idx[src] == placement_idx[tgt]
                    if not colocated:
                        continue

                    if mode == "full":
                        total_interference += float(vals.get("expected", 0.0))
                    elif mode == "no_probs":
                        total_interference += float(vals.get("stig", 0.0))
                    elif mode == "no_stig":
                        factor = float(vals.get("pn", 0.0)) + float(vals.get("ps", 0.0)) - float(vals.get("pns", 0.0))
                        total_interference += factor
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

        total_latency = compute_latency(target_services, latency_ms, placement_idx, comm_edges)

        return total_interference, total_latency

    return evaluate


# =========================
#   Printing helpers
# =========================

def print_best_solution(
    title: str,
    study: optuna.Study,
    target_services: List[Tuple[str, str]],
):
    print(f"\n=== {title} ===")
    if not study.best_trials:
        print("No completed trials.")
        return
    best = study.best_trials[0]

    placement_idx: Dict[Tuple[str, str], int] = {}
    for svc in target_services:
        key_str = key_to_str(svc)
        placement_idx[svc] = best.params[key_str]

    placement_named = {svc: NODE_NAMES[idx] for svc, idx in placement_idx.items()}

    for svc, node in placement_named.items():
        print(f"{pretty(svc):30s} -> {node}")

    print(f"\nTotal Interference: {best.values[0]:.2f}")
    print(f"Total Latency     : {best.values[1]:.2f} ms")

    print("\n--- Per-node ---")
    nodes = {n: [] for n in NODE_NAMES}
    for svc, idx in placement_idx.items():
        nodes[NODE_NAMES[idx]].append(svc)
    for node, svcs in nodes.items():
        print(node + ":")
        for s in sorted(svcs):
            print("  -", pretty(s))


# =========================
#   Pareto plot
# =========================

def plot_pareto(study: optuna.Study, filename: str = "pareto_front.png") -> None:
    """
    Plot the Pareto front for a 2-objective Optuna study:
      - Objective 0: total interference  (lower is better)
      - Objective 1: total latency       (lower is better)
    Saves the figure as `filename`.
    """
    trials = [t for t in study.trials if t.values is not None]
    if not trials:
        print("[plot_pareto] No completed trials; nothing to plot.")
        return

    interferences = np.array([t.values[0] for t in trials], dtype=float)
    latencies = np.array([t.values[1] for t in trials], dtype=float)

    # Approximate Pareto front: sort by interference and sweep minimum latency
    sorted_idx = np.argsort(interferences)
    inter_sorted = interferences[sorted_idx]
    lat_sorted = latencies[sorted_idx]

    pareto_x = []
    pareto_y = []
    best_lat = float("inf")
    for x, y in zip(inter_sorted, lat_sorted):
        if y < best_lat:
            pareto_x.append(x)
            pareto_y.append(y)
            best_lat = y
    pareto_x = np.array(pareto_x)
    pareto_y = np.array(pareto_y)

    # Pick a canonical "best" (min interference, then latency)
    best_trial = None
    best_vals = (float("inf"), float("inf"))
    for t in trials:
        inter, lat = t.values
        if (inter < best_vals[0]) or (inter == best_vals[0] and lat < best_vals[1]):
            best_vals = (inter, lat)
            best_trial = t

    plt.figure(figsize=(6, 4.5))
    plt.scatter(interferences, latencies, alpha=0.5, label="All trials")
    if len(pareto_x) > 0:
        plt.plot(pareto_x, pareto_y, linewidth=2.0, label="Pareto front")

    if best_trial is not None:
        bx, by = best_trial.values
        plt.scatter([bx], [by], s=80, marker="o", edgecolors="k",
                    label=f"Selected best ({bx:.1f}, {by:.1f})")

    plt.xlabel("Total interference (lower is better)")
    plt.ylabel("Total latency (ms, lower is better)")
    plt.title("Pareto front: interference–latency trade-off")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[plot_pareto] Saved Pareto plot to {filename}")
    plt.close()


# =========================
#   main
# =========================

def main():
    target_services, latency_ms, interference, comm_edges = build_problem()

    trial_label = trial_label_from_filename(PNS_CSV, default="")
    title_suffix = f" {trial_label}" if trial_label else ""

    print("\n=== Nodes ===", NODE_NAMES)
    print("=== STIG dir ===", STIK_DIR)
    print(f"Targets: {len(target_services)} | Pairs: {sum(len(v) for v in interference.values())}")

    print("\nMedian p95 latencies (ms):")
    for s in sorted(target_services):
        print(f"  {pretty(s):30s} {latency_ms.get(s, float('nan')):10.2f}")

    # ---- priorities (based on full expected) ----
    priorities = compute_service_priorities(target_services, interference)
    print("\n=== Service priorities (higher = more interference-sensitive) ===")
    for svc, score in sorted(priorities.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {pretty(svc):30s} score={score:.3f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    random.seed(42)
    sampler = optuna.samplers.NSGAIISampler(seed=42)

    modes = [
        ("full", "Causal (PN/PS/PNS × STIG)"),
        ("no_probs", "No PN/PS/PNS (STIG only)"),
        ("no_stig", "No STIG (PN/PS/PNS only)"),
        ("latency_only", "Latency-only (non-causal baseline)"),
    ]

    studies: Dict[str, optuna.Study] = {}

    for mode, label in modes:
        print(f"\n\n##### Optimising mode: {mode} – {label} #####")
        study = optuna.create_study(directions=["minimize", "minimize"],
                                    sampler=sampler)
        study.optimize(
            make_objective_variant(target_services, latency_ms, interference, comm_edges, mode),
            n_trials=200,
        )
        studies[mode] = study

        print_best_solution(f"Best Placement [{label}]{title_suffix}",
                            study, target_services)

        tag = trial_label or "noTag"
        plot_pareto(study, filename=f"pareto_front_{tag}_{mode}.png")

    # ---- summary table ----
    print("\n\n=== Ablation summary (best trial values) ===")
    for mode, label in modes:
        st = studies[mode]
        if not st.best_trials:
            continue
        best = st.best_trials[0]
        inter, lat = best.values
        print(f"{label:35s} | Interference={inter:10.3f} | Latency={lat:10.3f} ms")


if __name__ == "__main__":
    main()


