#!/usr/bin/env bash
# run-sglang-vision.sh — VISION + DSpark: ~150k ctx, ~150-200 tok/s, ~30 GB VRAM.
#
# Same SGLang + NVFP4 (LMHead4) + DSpark + flashinfer SM120 stack as godspeed,
# but the vision tower stays LIVE (no --language-only) and VRAM is rebalanced so
# the BF16 vision encoder + DSpark drafter both fit:
#   - --context-length 150000 / --max-total-tokens 150000   (was 237568)
#   - --mem-fraction-static 0.88                             (was 0.985)
#   - --speculative-dspark-block-size 5                      (was 7)  -> fits VRAM
#   - --mm-feature-transport cpu  (vision ON; avoids WSL2 CUDA-IPC quirk, harmless on Linux)
#
# Why text-only (godspeed) is "faster" here: weights are the same; the speed gap
# comes from block-size 7 + 98.5% VRAM (no vision). Cutting ctx is a VRAM-cost
# effect, not a speed cause — you only cut ctx to fit the vision tower.
#
# Usage:  ./run-sglang-vision.sh [start|stop|logs|status]   (KEEP_GPU=1 to not free the GPU)
# API:     http://localhost:${HOST_PORT}/v1   (also /anthropic)
set -uo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[ -f "$DIR/.env" ] && source "$DIR/.env"

SGGLANG_IMAGE="${SGGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
CONTAINER="${CONTAINER_NAME:-sglang-qwen38}"
HOST_PORT="${HOST_PORT:-8040}"
CONTAINER_PORT="${CONTAINER_PORT:-30000}"
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen3.8-27B-NVFP4-RTX5090-LMHead4}"
DRAFTER_SUBDIR="${DRAFTER_SUBDIR:-Qwen3.8-27B-DSpark-NVFP4}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-nvfp4}"
# Qwen3.8 chat-template knobs (JSON): reasoning_effort xhigh/medium/low,
# enable_thinking, preserve_thinking. Unset -> default reasoning_effort
# medium (balances accuracy and speed). Override in .env, e.g.
#   DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"xhigh"}'
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ] || DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium"}'
VLLM_COMPOSE_DIR="${VLLM_COMPOSE_DIR:-}"

MODEL_HOST="$DIR/models/$MODEL_SUBDIR"
DRAFTER_HOST="$DIR/models/$DRAFTER_SUBDIR"

require_model() {
  [ -f "$MODEL_HOST/config.json" ] || {
    echo "ERROR: target model not at $MODEL_HOST"; echo "  Run ./setup.sh weights first."; exit 1; }
  [ -d "$DRAFTER_HOST" ] || {
    echo "ERROR: DSpark drafter not at $DRAFTER_HOST"; echo "  Run ./setup.sh weights first."; exit 1; }
}

start() {
  require_model
  podman image exists "$SGGLANG_IMAGE" || {
    echo "ERROR: image $SGGLANG_IMAGE missing — run: podman pull $SGGLANG_IMAGE"; exit 1; }

  if [ "${KEEP_GPU:-0}" != "1" ]; then
    if [ -n "$VLLM_COMPOSE_DIR" ] && [ -f "$VLLM_COMPOSE_DIR/docker-compose.yml" ]; then
      echo "[gpu] stopping the whole vLLM compose project in $VLLM_COMPOSE_DIR (no -v)"
      ( cd "$VLLM_COMPOSE_DIR" && podman-compose -f docker-compose.yml down 2>/dev/null || true )
    else
      echo "[gpu] freeing GPU: stopping any running sglang/vllm/llama containers that own the 5090 ..."
      for c in sglang-qwen38 vllm-qwen38 qwen38-vllm; do
        if podman ps -q -f name="$c" 2>/dev/null | grep -q .; then
          echo "  [gpu] stopping $c"; podman stop "$c" >/dev/null 2>&1 || true
        fi
      done
      echo "  [gpu] TIP: set VLLM_COMPOSE_DIR=/path/to/vllm-compose in .env for a clean stop"
      echo "        (vLLM's restart:always makes plain 'podman stop' get undone)."
    fi
    sleep 3
  fi

  local memfree
  memfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[gpu] free MiB: ${memfree:-?}"
  if [ -n "${memfree:-}" ] && [ "$memfree" -lt 20000 ] 2>/dev/null; then
    echo "WARNING: only ${memfree} MiB free — SGLang needs ~30 GB on the 5090. Free the GPU or use KEEP_GPU=1."
  fi

  echo "Starting SGLang (VISION + DSpark) -> http://localhost:${HOST_PORT}/v1"
  echo "  target : $MODEL_HOST"
  echo "  drafter: $DRAFTER_HOST"
  echo "  (vision ON; no --language-only)"
  podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  # Join the monitor network so Prometheus (docker-compose) can scrape /metrics
  # as sglang-qwen38:30000. Idempotent: podman ignores an already-attached net.
  podman network inspect sglang_monitor >/dev/null 2>&1 || podman network create sglang_monitor >/dev/null 2>&1 || true
  podman run -d \
    --name "$CONTAINER" --replace \
    --network sglang_monitor \
    --device nvidia.com/gpu=all \
    --ipc host \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    -v "$MODEL_HOST:/model:ro" \
    -v "$DRAFTER_HOST:/model_dspark:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e MAX_JOBS=4 \
    -e NVCC_THREADS=4 \
    -e "SGLANG_API_KEY=${API_KEY}" \
    --entrypoint python3 \
    "$SGGLANG_IMAGE" \
    -m sglang.launch_server \
      --model-path /model \
      --served-model-name "$SERVED_MODEL_NAME" \
      --trust-remote-code \
      --tp-size 1 \
      --host 0.0.0.0 --port "$CONTAINER_PORT" \
      --kv-cache-dtype fp8_e4m3 \
      --attention-backend flashinfer \
      --context-length 150000 \
      --chunked-prefill-size 2048 \
      --mamba-radix-cache-strategy extra_buffer_lazy \
      --mamba-ssm-dtype bfloat16 \
      --max-mamba-cache-size 5 \
      --mem-fraction-static 0.82 \
      --max-running-requests 1 \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path /model_dspark \
      --speculative-dspark-block-size 5 \
      --speculative-draft-model-quantization modelopt_fp4 \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
      --tool-call-parser qwen3_coder \
      --mm-feature-transport cpu \
      --enable-metrics \
    2>&1 | tail -20

  echo "Container started. First boot loads ~18 GB + vision tower + drafter (~30 GB)."
  echo "  If it OOMs at boot: lower --mem-fraction-static to 0.80 and --context-length to ~120000, then re-run."
  echo "Watch readiness: ./run-sglang-vision.sh status"
}

stop()  { podman rm -f "$CONTAINER" >/dev/null 2>&1 && echo "stopped" || echo "not running"; }
logs()  { podman logs -f "$CONTAINER"; }
status(){
  echo "--- container ---"; podman ps -a --filter name="$CONTAINER" --format '{{.Names}}  {{.Status}}'
  echo "--- api health ---"
  curl -s -o /dev/null -w "HTTP %{http_code}  (200 = ready)\n" "http://localhost:${HOST_PORT}/health" || echo "not responding yet"
  curl -s "http://localhost:${HOST_PORT}/v1/models" 2>/dev/null | head -c 300; echo
  echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
}

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  logs)  logs ;;
  status) status ;;
  *) echo "usage: $0 [start|stop|logs|status]  (KEEP_GPU=1 to not free the GPU)"; exit 2 ;;
esac
