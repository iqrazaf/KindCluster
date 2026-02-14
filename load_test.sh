#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# load_test.sh
# - Sets up port-forwarding for:
#     * Istio ingressgateway      -> localhost:${INGRESS_LOCAL_PORT:-8080}
#     * Prometheus                -> localhost:${PROM_LOCAL_PORT:-9090}
#     * Jaeger (trace backend)    -> localhost:${TRACE_LOCAL_PORT:-16686}  (optional)
# - Installs k6 if missing (Linux)
# - Writes the spike-load script (s1_spike.js)
# - Runs k6 spike test against your KindCluster ingress:
#     http://localhost:<INGRESS_LOCAL_PORT>/sock
#     http://localhost:<INGRESS_LOCAL_PORT>/tea
#     http://localhost:<INGRESS_LOCAL_PORT>/book
# - Exports a JSON summary you can archive/compare
# - Optionally dumps trace data from Jaeger HTTP API
# ============================================================

ISTIO_NS="${ISTIO_NS:-istio-system}"
INGRESS_SVC="${INGRESS_SVC:-istio-ingressgateway}"
INGRESS_LOCAL_PORT="${INGRESS_LOCAL_PORT:-8080}"

PROM_SVC="${PROM_SVC:-prometheus}"
PROM_LOCAL_PORT="${PROM_LOCAL_PORT:-9090}"

TRACE_SVC="${TRACE_SVC:-jaeger-query}"     # override if you use a different tracing svc name
TRACE_LOCAL_PORT="${TRACE_LOCAL_PORT:-16686}"

BASE_URL="${BASE_URL:-http://127.0.0.1:${INGRESS_LOCAL_PORT}}"
OUTDIR="${OUTDIR:-load_results}"
OUTJSON="${OUTJSON:-${OUTDIR}/spike_summary_$(date +%Y%m%d_%H%M%S).json}"

mkdir -p "${OUTDIR}"

log() { printf "\n[INFO] %s\n" "$*"; }

PF_PIDS=()

cleanup() {
  if ((${#PF_PIDS[@]} > 0)); then
    log "Cleaning up port-forward processes"
    for pid in "${PF_PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT

require_kubectl() {
  if ! command -v kubectl >/dev/null 2>&1; then
    echo "[ERR ] kubectl not found in PATH; install kubectl and ensure kube-context points to your Kind cluster."
    exit 1
  fi
}

start_port_forward() {
  local ns="$1"
  local svc="$2"
  local local_port="$3"
  local remote_port="$4"
  local label="$5"

  if ! kubectl -n "${ns}" get svc "${svc}" >/dev/null 2>&1; then
    log "Skipping port-forward for ${label}: service '${svc}' not found in namespace '${ns}'."
    return
  fi

  log "Port-forwarding ${label}: localhost:${local_port} -> ${svc}.${ns}:${remote_port}"
  kubectl -n "${ns}" port-forward "svc/${svc}" "${local_port}:${remote_port}" >/dev/null 2>&1 &
  local pf_pid=$!
  PF_PIDS+=("${pf_pid}")
  # give it a moment to establish the tunnel
  sleep 3
}

install_k6_linux() {
  if command -v k6 >/dev/null 2>&1; then
    return
  fi

  if [[ "$(uname -s | tr '[:upper:]' '[:lower:]')" != "linux" ]]; then
    echo "[ERR ] k6 not found. Install k6 manually (https://k6.io/docs/get-started/installation/) and re-run."
    exit 1
  fi

  log "k6 not found. Installing k6 (Linux) via apt repo..."
  sudo gpg -k >/dev/null 2>&1 || true
  sudo apt-get update -y
  sudo apt-get install -y ca-certificates gnupg curl

  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://dl.k6.io/key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/k6-archive-keyring.gpg
  echo "deb [signed-by=/etc/apt/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
    sudo tee /etc/apt/sources.list.d/k6.list >/dev/null

  sudo apt-get update -y
  sudo apt-get install -y k6
}

write_k6_script() {
  cat > s1_spike.js <<'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomSeed } from 'k6';
randomSeed(42);

export const options = {
  summaryTrendStats: ['avg','min','med','p(90)','p(95)','max'],
  scenarios: {
    spike_all: {
      executor: 'ramping-arrival-rate',
      startRate: 10, // warm start
      timeUnit: '1s',
      preAllocatedVUs: 200,
      maxVUs: 400,
      stages: [
        { target: 30,  duration: '60s' }, // warm up
        { target: 400, duration: '20s' }, // sudden spike
        { target: 400, duration: '2m'  }, // hold
        { target: 60,  duration: '60s' }, // ramp down
        { target: 0,   duration: '30s' }, // drain
      ],
    },
  },
  thresholds: {
    'http_req_failed': ['rate<0.02'], // <2% failures
    'http_req_duration{app:sock}': ['p(95)<1000'],
    'http_req_duration{app:book}': ['p(95)<1000'],
    'http_req_duration{app:tea}':  ['p(95)<1500'],
  },
};

// BASE_URL is injected from the environment (load_test.sh)
// Defaults to localhost:INGRESS_LOCAL_PORT if not set.
const BASE = __ENV.BASE_URL || 'http://127.0.0.1:8080';

const TARGETS = [
  { path: '/sock',              tag: 'sock', weight: 0.55 },
  { path: '/book?u=normal',     tag: 'book', weight: 0.20 },
  { path: '/book?u=test',       tag: 'book', weight: 0.05 },
  { path: '/tea',               tag: 'tea',  weight: 0.20 },
];

const totalW = TARGETS.reduce((a, b) => a + b.weight, 0);

function pick() {
  let r = Math.random() * totalW;
  for (const t of TARGETS) {
    r -= t.weight;
    if (r <= 0) return t;
  }
  return TARGETS[0];
}

export default function () {
  const t = pick();
  const res = http.get(`${BASE}${t.path}`, { tags: { app: t.tag } });
  check(res, { 'status 2xx/3xx': r => r.status >= 200 && r.status < 400 });
  sleep(Math.random() * 0.8);
}
EOF
}

dump_traces() {
  # Optional Jaeger/trace dump – best-effort only
  local trace_file="${OUTDIR}/traces_$(date +%Y%m%d_%H%M%S).json"

  log "Attempting to dump trace data from Jaeger (if available)..."
  if curl -fsS "http://127.0.0.1:${TRACE_LOCAL_PORT}/api/traces?limit=200&lookback=1h" -o "${trace_file}"; then
    log "Trace data saved to: ${trace_file}"
  else
    log "Trace dump skipped (Jaeger not reachable on localhost:${TRACE_LOCAL_PORT} or API not available)."
    rm -f "${trace_file}" || true
  fi
}

main() {
  log "Checking kubectl and cluster access"
  require_kubectl

  log "Ensuring k6 is installed"
  install_k6_linux

  log "Starting port-forwarding for ingress / Prometheus / tracing (if present)"
  start_port_forward "${ISTIO_NS}" "${INGRESS_SVC}"  "${INGRESS_LOCAL_PORT}" 80   "Istio ingressgateway"
  start_port_forward "${ISTIO_NS}" "${PROM_SVC}"     "${PROM_LOCAL_PORT}"  9090 "Prometheus"
  start_port_forward "${ISTIO_NS}" "${TRACE_SVC}"    "${TRACE_LOCAL_PORT}" 16686 "Jaeger (tracing backend)"

  log "Writing k6 script: s1_spike.js"
  write_k6_script

  log "Quick sanity: showing first lines of s1_spike.js"
  head -20 s1_spike.js

  log "Running spike test against BASE_URL=${BASE_URL}"
  export BASE_URL
  k6 run --summary-export "${OUTJSON}" s1_spike.js

  log "k6 run complete. Summary exported to: ${OUTJSON}"

  # Optional: best-effort trace dump from Jaeger
  dump_traces

  log "Tip: start with a lower peak if the VM is small (edit targets 400 -> 150–200 in s1_spike.js)."
}

main "$@"
