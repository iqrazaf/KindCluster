#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# load_test.sh
# - Installs k6 if missing (Linux)
# - Writes the spike-load script (s1_spike.js)
# - Runs k6 spike test against your KindCluster ingress:
#     http://localhost/sock
#     http://localhost/tea
#     http://localhost/book
# - Exports a JSON summary you can archive/compare
# ============================================================

BASE_URL="${BASE_URL:-http://127.0.0.1}"
OUTDIR="${OUTDIR:-load_results}"
OUTJSON="${OUTJSON:-${OUTDIR}/spike_summary_$(date +%Y%m%d_%H%M%S).json}"

mkdir -p "${OUTDIR}"

log() { printf "\n[INFO] %s\n" "$*"; }

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
  sudo apt-get install -y ca-certificates gnupg

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

// Default to hostPort 80 on the kind VM
const BASE = __ENV.BASE_URL || 'http://127.0.0.1';

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

main() {
  log "Installing/Checking k6"
  install_k6_linux

  log "Writing k6 script: s1_spike.js"
  write_k6_script

  log "Quick sanity: showing first lines"
  head -20 s1_spike.js

  log "Running spike test against BASE_URL=${BASE_URL}"
  export BASE_URL
  k6 run --summary-export "${OUTJSON}" s1_spike.js

  log "Done. Summary exported to: ${OUTJSON}"
  log "Tip: start lower peak if VM is small (edit targets 400 -> 150–200)."
}

main "$@"
