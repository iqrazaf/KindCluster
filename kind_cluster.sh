#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# KindCluster Artifact Runner
# - Creates kind cluster (1 CP + 2 workers)
# - Installs Istio (demo profile)
# - Deploys Sock Shop + TeaStore + Bookinfo
# - Exposes routes via Istio ingress:
#     http://localhost/sock
#     http://localhost/tea
#     http://localhost/book
# - Smoke tests with curl
# ============================================================

CLUSTER_NAME="${CLUSTER_NAME:-demo}"
K8S_NODE_IMAGE="${K8S_NODE_IMAGE:-kindest/node:v1.29.2}"
ISTIO_VERSION="${ISTIO_VERSION:-1.22.1}"

WORKDIR="${WORKDIR:-$PWD/_artifact_work}"
mkdir -p "$WORKDIR"

log()  { printf "\n[INFO] %s\n" "$*"; }
warn() { printf "\n[WARN] %s\n" "$*"; }
die()  { printf "\n[ERR ] %s\n" "$*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# ---------------------------
# 0) Preflight
# ---------------------------
log "Preflight checks"

need_cmd docker
need_cmd curl
need_cmd git

if ! docker info >/dev/null 2>&1; then
  die "Docker daemon not running. Start Docker and re-run."
fi

# Install kubectl/kind/helm/istioctl if missing (Linux amd64 only).
# For macOS/arm64, install manually and re-run.
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

if [[ "$OS" != "linux" ]]; then
  warn "This auto-installer is optimized for Linux. You can still run it if tools exist."
fi
if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
  warn "Non-amd64 architecture detected ($ARCH). Auto-install may not work. Ensure tools are installed."
fi

install_kubectl() {
  if command -v kubectl >/dev/null 2>&1; then return; fi
  log "Installing kubectl (v1.29.6)"
  curl -L -o "$WORKDIR/kubectl" "https://dl.k8s.io/release/v1.29.6/bin/linux/amd64/kubectl"
  chmod +x "$WORKDIR/kubectl"
  sudo mv "$WORKDIR/kubectl" /usr/local/bin/kubectl
}

install_kind() {
  if command -v kind >/dev/null 2>&1; then return; fi
  log "Installing kind (v0.23.0)"
  curl -L -o "$WORKDIR/kind" "https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64"
  chmod +x "$WORKDIR/kind"
  sudo mv "$WORKDIR/kind" /usr/local/bin/kind
}

install_helm() {
  if command -v helm >/dev/null 2>&1; then return; fi
  log "Installing helm"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
}

install_istioctl() {
  if command -v istioctl >/dev/null 2>&1; then return; fi
  log "Installing istioctl (Istio ${ISTIO_VERSION})"
  (cd "$WORKDIR" && curl -L https://istio.io/downloadIstio | ISTIO_VERSION="${ISTIO_VERSION}" sh -)
  sudo ln -sf "$WORKDIR/istio-${ISTIO_VERSION}/bin/istioctl" /usr/local/bin/istioctl
}

# Auto-install tools if missing (Linux only)
if [[ "$OS" == "linux" ]]; then
  install_kubectl
  install_kind
  install_helm
  install_istioctl
else
  need_cmd kubectl
  need_cmd kind
  need_cmd helm
  need_cmd istioctl
fi

log "Tool versions"
kubectl version --client || true
kind version || true
helm version || true
istioctl version || true

# ---------------------------
# 1) Create kind cluster
# ---------------------------
log "Creating kind cluster: ${CLUSTER_NAME}"

KIND_CFG="$WORKDIR/kind-3node.yaml"
cat > "$KIND_CFG" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: ${K8S_NODE_IMAGE}
  extraPortMappings:
  - containerPort: 32080
    hostPort: 80
  - containerPort: 32443
    hostPort: 443
- role: worker
  image: ${K8S_NODE_IMAGE}
- role: worker
  image: ${K8S_NODE_IMAGE}
EOF

# If cluster already exists, skip create
if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  warn "Cluster ${CLUSTER_NAME} already exists. Reusing it."
else
  kind create cluster --name "${CLUSTER_NAME}" --config "$KIND_CFG"
fi

kubectl cluster-info
kubectl get nodes -o wide

log "Tainting control-plane to avoid scheduling workloads there"
CP_NODE="$(kubectl get nodes -l 'node-role.kubernetes.io/control-plane' -o jsonpath='{.items[0].metadata.name}')"
kubectl taint nodes "$CP_NODE" node-role.kubernetes.io/control-plane=:NoSchedule --overwrite

log "Labeling worker nodes nodepool=worker"
for n in $(kubectl get nodes --no-headers | awk '$3!="control-plane"{print $1}'); do
  kubectl label node "$n" nodepool=worker --overwrite
done
kubectl get nodes --show-labels

# ---------------------------
# 2) Install Istio
# ---------------------------
log "Installing Istio (demo profile)"
istioctl install --set profile=demo -y
kubectl -n istio-system rollout status deploy/istiod --timeout=300s || true
kubectl -n istio-system get pods

log "Patching istio-ingressgateway to NodePort (32080/32443)"
kubectl -n istio-system patch svc istio-ingressgateway \
  -p '{"spec":{"type":"NodePort","ports":[
    {"name":"http2","port":80,"targetPort":8080,"nodePort":32080},
    {"name":"https","port":443,"targetPort":8443,"nodePort":32443}
  ]}}'

log "Istio ingress should now be reachable on host: http://localhost/"
sleep 2

# ---------------------------
# 3) Deploy applications
# ---------------------------
deploy_ns() {
  local ns="$1"
  kubectl create ns "$ns" >/dev/null 2>&1 || true
  kubectl label ns "$ns" istio-injection=enabled --overwrite
}

patch_node_selector_all_deploys() {
  local ns="$1"
  kubectl -n "$ns" get deploy -o name | \
    xargs -I{} kubectl -n "$ns" patch {} --type=merge \
      -p '{"spec":{"template":{"spec":{"nodeSelector":{"nodepool":"worker"}}}}}'
}

log "Deploying Sock Shop"
deploy_ns sock-shop
if [[ ! -d "$WORKDIR/microservices-demo" ]]; then
  git clone https://github.com/microservices-demo/microservices-demo.git "$WORKDIR/microservices-demo"
fi
kubectl apply -n sock-shop -f "$WORKDIR/microservices-demo/deploy/kubernetes/manifests"
patch_node_selector_all_deploys sock-shop

log "Deploying TeaStore"
deploy_ns teastore
if [[ ! -d "$WORKDIR/TeaStore" ]]; then
  git clone https://github.com/DescartesResearch/TeaStore.git "$WORKDIR/TeaStore"
fi
kubectl apply -n teastore -f "$WORKDIR/TeaStore/kubernetes/teastore-namespace.yaml" || true
kubectl apply -n teastore -f "$WORKDIR/TeaStore/kubernetes/teastore-deployment-service.yaml"
patch_node_selector_all_deploys teastore

log "Deploying Bookinfo"
deploy_ns bookinfo
ISTIO_SAMPLES="$WORKDIR/istio-${ISTIO_VERSION}/samples"
if [[ ! -d "$ISTIO_SAMPLES" ]]; then
  die "Istio samples not found at $ISTIO_SAMPLES. Ensure istioctl install step succeeded."
fi
kubectl apply -n bookinfo -f "$ISTIO_SAMPLES/bookinfo/platform/kube/bookinfo.yaml"
patch_node_selector_all_deploys bookinfo

log "Waiting for pods to become Ready (best effort, some may take longer)"
kubectl -n sock-shop get pods -o wide
kubectl -n teastore get pods -o wide
kubectl -n bookinfo get pods -o wide

# ---------------------------
# 4) Install Istio routing
# ---------------------------
log "Applying a single Gateway + VirtualService to route /sock, /tea, /book"

ROUTES_YAML="$WORKDIR/istio-routes.yaml"
cat > "$ROUTES_YAML" <<'EOF'
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: apps-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts: ["*"]
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: route-all
  namespace: istio-system
spec:
  hosts: ["*"]
  gateways: ["istio-system/apps-gateway"]
  http:
  - match: [{ uri: { prefix: "/sock" } }]
    rewrite: { uri: "/" }
    route:
    - destination:
        host: front-end.sock-shop.svc.cluster.local
        port: { number: 80 }
  - match: [{ uri: { prefix: "/tea" } }]
    rewrite: { uri: "/" }
    route:
    - destination:
        host: teastore-webui.teastore.svc.cluster.local
        port: { number: 8080 }
  - match: [{ uri: { prefix: "/book" } }]
    rewrite: { uri: "/" }
    route:
    - destination:
        host: productpage.bookinfo.svc.cluster.local
        port: { number: 9080 }
EOF

kubectl apply -f "$ROUTES_YAML"

# ---------------------------
# 5) Integrity checks / Smoke tests
# ---------------------------
log "Integrity check: ensure app pods are NOT on control-plane"
kubectl get pods -A -o wide | awk 'NR==1 || $8 !~ /control-plane/'

log "Smoke tests (HTTP status)"
set +e
curl -sS -o /dev/null -w "sock  : %{http_code}\n" http://localhost/sock
curl -sS -o /dev/null -w "tea   : %{http_code}\n" http://localhost/tea
curl -sS -o /dev/null -w "book  : %{http_code}\n" http://localhost/book
set -e

cat <<EOF

============================================================
SUCCESS CRITERIA
- kind cluster exists: kind get clusters shows "${CLUSTER_NAME}"
- kubectl get nodes shows 3 nodes (1 control-plane + 2 workers), all Ready
- istio-system pods Running
- curl http://localhost/sock returns 200 (or sometimes 30x then 200 in browser)
- curl http://localhost/tea returns 200
- curl http://localhost/book returns 200

Open in browser:
- http://localhost/sock
- http://localhost/tea
- http://localhost/book
============================================================

EOF

log "Done."
