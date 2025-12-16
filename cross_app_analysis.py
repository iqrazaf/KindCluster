import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ---------------- CONFIG ----------------
CSV_PATH = Path("trials_clean_traces/clean_traces_T3.csv")
OUT_CSV = Path("Probability_Table/cross_shop_PNS_T3.csv")

DELTA_T = 60          # seconds: Y must start before (X_end + DELTA_T)
Z_THRESHOLD = 1.0     # duration anomaly: mean + Z*std
MIN_EVENTS_PER_SERVICE = 5  # filter very tiny series
# ----------------------------------------

APP_OF = {
    # TeaStore
    "teastore-webui": "teastore", "teastore-auth": "teastore",
    "teastore-registry": "teastore", "teastore-persistence": "teastore",
    "teastore-image": "teastore",
    # BookInfo
    "productpage": "bookinfo", "details": "bookinfo",
    "reviews": "bookinfo", "ratings": "bookinfo",
    # Sock-Shop
    "front-end": "sock-shop", "queue-master": "sock-shop", "payment": "sock-shop",
    "user": "sock-shop", "catalogue": "sock-shop", "orders": "sock-shop",
    "carts": "sock-shop", "shipping": "sock-shop"
}


def parse_time(s: str):
    s = str(s).strip()
    fmts = ["%I:%M:%S %p", "%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    for f in fmts:
        try:
            base = datetime.today().date()
            t = datetime.strptime(s, f)
            return datetime.combine(base, t.time())
        except ValueError:
            continue
    # last resort: let pandas try
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return pd.NaT


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        usecols=["spanID", "traceID", "service_name", "startTime_str", "duration", "error_tag"]
    )
    # normalize service id (strip namespace)
    df["service_name"] = df["service_name"].astype(str).str.strip().str.split(".").str[0]
    df["app"] = df["service_name"].map(APP_OF).fillna("unknown")

    df["ts"] = df["startTime_str"].apply(parse_time)
    df = df.dropna(subset=["ts"]).copy()

    # duration in seconds (float), fall back to 0
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["t_end"] = df["ts"] + pd.to_timedelta(df["duration"], unit="s")

    # error flag (bool)
    df["error_tag"] = df["error_tag"].astype(str).str.lower().isin(["true", "1", "yes"])

    print(f"Loaded {len(df)} spans across {df['service_name'].nunique()} services.")
    return df


def mark_anomalies(df: pd.DataFrame, z: float) -> pd.DataFrame:
    """Anomaly = error_tag == True OR duration > (mean + z*std) per service."""
    stats = df.groupby("service_name")["duration"].agg(["mean", "std"]).fillna(0.0)
    thr = stats["mean"] + z * stats["std"]
    df = df.copy()
    df["dur_thr"] = df["service_name"].map(thr.to_dict())
    df["anomalous"] = df["error_tag"] | (df["duration"] > df["dur_thr"])

    print("\n=== Per-service duration stats (mean, std, threshold) ===")
    print(pd.concat([stats, thr.rename("thr")], axis=1).round(3))
    print(f"\nTotal anomalous spans (error OR duration spike): {int(df['anomalous'].sum())}")
    return df


def _temporal_hits(x_ts: np.ndarray, x_te: np.ndarray, y_ts: np.ndarray, delta_t: float) -> int:
    """
    Count how many X-events have at least one Y-event that starts in [x_ts, x_te + delta].
    Uses NumPy; no pandas reductions over datetime dtypes.
    """
    hits = 0
    delta = np.timedelta64(int(delta_t * 1e3), 'ms')  # convert seconds to ms timedelta
    for i in range(x_ts.shape[0]):
        start = x_ts[i]
        end_plus = x_te[i] + delta
        # boolean mask over y_ts (numpy datetime64)
        m = (y_ts >= start) & (y_ts <= end_plus)
        if m.any():
            hits += 1
    return hits


def _independent_fraction(x_ts: np.ndarray, x_te: np.ndarray, y_ts: np.ndarray, delta_t: float) -> float:
    """
    Fraction of Y-events that do NOT fall into any (x_ts, x_te + delta) window.
    """
    if y_ts.size == 0:
        return 0.0
    delta = np.timedelta64(int(delta_t * 1e3), 'ms')
    independent = 0
    for t in y_ts:
        # is t covered by ANY X-window?
        covered = ((x_ts <= t) & (t <= (x_te + delta))).any()
        if not covered:
            independent += 1
    return independent / float(y_ts.size)


def compute_cross_app(df: pd.DataFrame, delta_t: float) -> pd.DataFrame:
    # Keep only anomalous spans (hybrid logic)
    dfa = df[df["anomalous"]].copy()
    if dfa.empty:
        print("\n[Info] No anomalous spans found under current rules.")
        return pd.DataFrame()

    # Minimum events per service filter
    cnt = dfa.groupby("service_name").size()
    keep_svcs = cnt[cnt >= MIN_EVENTS_PER_SERVICE].index.tolist()
    dfa = dfa[dfa["service_name"].isin(keep_svcs)].copy()
    if dfa["service_name"].nunique() < 2:
        print("\n[Info] Only one service left after filtering; need cross-shop anomalies.")
        return pd.DataFrame()

    # group by service for speed
    groups = {s: g.sort_values("ts") for s, g in dfa.groupby("service_name")}

    results = []
    for y, gy in groups.items():
        y_app = APP_OF.get(y, "unknown")
        y_ts = gy["ts"].to_numpy(dtype="datetime64[ns]")
        # x loops over other services in different apps
        for x, gx in groups.items():
            if x == y:
                continue
            x_app = APP_OF.get(x, "unknown")
            if x_app == y_app:
                continue

            x_ts = gx["ts"].to_numpy(dtype="datetime64[ns]")
            x_te = gx["t_end"].to_numpy(dtype="datetime64[ns]")

            if x_ts.size == 0 or y_ts.size == 0:
                continue

            total_x = x_ts.size
            triggered = _temporal_hits(x_ts, x_te, y_ts, delta_t)
            p_y_given_x = triggered / total_x if total_x else 0.0
            p_y_given_notx = _independent_fraction(x_ts, x_te, y_ts, delta_t)

            pns = max(0.0, p_y_given_x - p_y_given_notx)
            pn = (pns / p_y_given_x) if p_y_given_x > 0 else 0.0
            ps = (pns / (1.0 - p_y_given_notx)) if (1.0 - p_y_given_notx) > 0 else 0.0

            results.append({
                "X": x, "Y": y, "App(X)": x_app, "App(Y)": y_app,
                "P(Y|X)": round(p_y_given_x * 100, 2),
                "P(Y|X')": round(p_y_given_notx * 100, 2),
                "PNS": round(pns * 100, 2),
                "PN": round(pn * 100, 2),
                "PS": round(ps * 100, 2),
                "X_events": int(total_x), "Y_events": int(y_ts.size)
            })

    if not results:
        print("\n[Info] No valid cross-shop interference pairs found (still single-app anomalies).")
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(["PN", "PNS", "PS"], ascending=False)


def main():
    df = load_data()
    df = mark_anomalies(df, Z_THRESHOLD)
    res = compute_cross_app(df, DELTA_T)

    if res.empty:
        print("\n[No Results] PN/PS/PNS table is empty. Likely only one shop shows anomalies "
              "even after duration-based detection. Collect traces_T1 with multi-shop stress.")
        return

    print(f"\n=== Cross-Shop PN/PS/PNS (Δt={DELTA_T}s; duration z>{Z_THRESHOLD}) ===")
    print(res.to_string(index=False))
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results → {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()







