#!/usr/bin/env bash
# monitor.sh — manage the SGLang monitoring stack (caddy/prometheus/grafana/dcgm).
# Isolated from the vLLM stack; uses docker-compose.yml (podman-compose) in this dir.
#
#   ./monitor.sh up       # podman-compose up -d  (builds dcgm-sglang-exporter on first run)
#   ./monitor.sh down     # stop + remove the monitoring containers (SGLang server unaffected)
#   ./monitor.sh status   # container states + Prometheus target health
#   ./monitor.sh logs     # tail all four service logs
#   ./monitor.sh dashboard# print the Grafana dashboard URL + login hints
set -uo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Fall back to bin/ shims (podman -> docker) when podman isn't installed.
PATH="$PATH:$DIR/bin"
[ -f "$DIR/.env" ] && source "$DIR/.env"

GRAFANA_HOST_PORT="${GRAFANA_HOST_PORT:-8042}"
PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-9091}"
GATEWAY_HOST_PORT="${GATEWAY_HOST_PORT:-8041}"

cmd="${1:-up}"
case "$cmd" in
  up)
    echo "Bringing up SGLang monitoring stack ..."
    # Build the (local, public-based) dcgm exporter image first, then compose up.
    podman build -t dcgm-sglang-exporter "$DIR/dcgm-exporter" >/dev/null 2>&1 || true
    ( cd "$DIR" && podman-compose -f docker-compose.yml up -d )
    echo "Up. Dashboard: http://localhost:${GRAFANA_HOST_PORT}  (Prometheus: 127.0.0.1:${PROMETHEUS_HOST_PORT})"
    ;;
  down)
    ( cd "$DIR" && podman-compose -f docker-compose.yml down )
    ;;
  status)
    echo "=== monitoring containers ==="
    podman ps -a --filter "name=caddy-sglang" --filter "name=prometheus-sglang" \
      --filter "name=grafana-sglang" --filter "name=dcgm-sglang" \
      --format '{{.Names}}\t{{.Status}}'
    echo "=== Prometheus targets ==="
    curl -s "http://127.0.0.1:${PROMETHEUS_HOST_PORT}/api/v1/targets" 2>/dev/null \
      | python3 -c "import sys,json
d=json.load(sys.stdin)
for t in d['data']['activeTargets']:
    print(f\"  {t['labels'].get('job','?'):8} {t['scrapeUrl']:32} -> {t['health']}\")" 2>/dev/null \
      || echo "  (Prometheus not reachable on 127.0.0.1:${PROMETHEUS_HOST_PORT})"
    ;;
  logs)
    for c in caddy-sglang prometheus-sglang grafana-sglang dcgm-sglang; do
      echo "===== $c ====="; podman logs --tail 30 "$c" 2>&1
    done
    ;;
  dashboard)
    echo "Grafana dashboard: http://localhost:${GRAFANA_HOST_PORT}/d/sglang-qwen38-27b"
    echo "  login: admin / GRAFANA_ADMIN_PASSWORD (from .env), or admin/admin if unset"
    echo "Gateway (Basic auth): http://localhost:${GATEWAY_HOST_PORT}  (METRICS_USER / METRICS_PASSWORD from .env)"
    ;;
  *)
    echo "usage: $0 [up|down|status|logs|dashboard]"
    exit 2
    ;;
esac
