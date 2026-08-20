#!/usr/bin/env bash
# run-sglang-godspeed.sh — GODSPEED recipe: TEXT-ONLY, ~236k ctx, ~180-260 tok/s.
# SGLang + NVFP4 (LMHead4) + DSpark drafter + flashinfer SM120 FP4 GEMM on the RTX 5090.
#
# This is the fast ceiling from HF discussion #11 (cosmicnag) + the official SGLang
# Qwen3.8-27B RTX-5090 cell (flashinfer backend, mamba state levers).
#
# TEXT-ONLY: --language-only is set, so the vision tower is OFF (frees VRAM for
# DSpark block-size 7 + the 236k pool). If you want vision, use run-sglang-vision.sh.
#
# Usage:
#   ./run-sglang-godspeed.sh start      # start (idempotent: replaces running container)
#   ./run-sglang-godspeed.sh stop
#   ./run-sglang-godspeed.sh logs
#   ./run-sglang-godspeed.sh status
#   KEEP_GPU=1 ./run-sglang-godspeed.sh start   # don't free the GPU
#
# API (OpenAI-compatible):  http://localhost:${HOST_PORT}/v1
# (Also Anthropic-compatible: /anthropic.)
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
# enable_thinking, preserve_thinking. Unset -> behavior-preserving default
# (thinking on, preserved, reasoning_effort xhigh). Override in .env, e.g.
#   DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium"}'
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ] || DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"xhigh"}'
# Optional: path to the vLLM compose dir (its docker-compose.yml) so we can stop
# the whole vLLM project cleanly. Its containers use `restart: always`, so a
# plain `podman stop` gets undone; `podman-compose down` removes them for good
# (no -v, so its monitoring data volumes are preserved).
VLLM_COMPOSE_DIR="${VLLM_COMPOSE_DIR:-}"

MODEL_HOST="$DIR/models/$MODEL_SUBDIR"
DRAFTER_HOST="$DIR/models/$DRAFTER_SUBDIR"

require_model() {
  [ -f "$MODEL_HOST/config.json" ] || {
    echo "ERROR: target model not at $MODEL_HOST"; echo "  Run ./setup.sh weights first (and set MODEL_REVISION if needed)."; exit 1; }
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

  echo "Starting SGLang (godspeed, text-only) -> http://localhost:${HOST_PORT}/v1"
  echo "  target : $MODEL_HOST"
  echo "  drafter: $DRAFTER_HOST"
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
      --context-length 237568 \
      --chunked-prefill-size 2048 \
      --mamba-radix-cache-strategy extra_buffer_lazy \
      --mamba-ssm-dtype bfloat16 \
      --max-mamba-cache-size 5 \
      --mem-fraction-static 0.90 \
      --max-running-requests 1 \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path /model_dspark \
      --speculative-dspark-block-size 7 \
      --speculative-draft-model-quantization modelopt_fp4 \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
      --tool-call-parser qwen3_coder \
      --mm-feature-transport cpu \
      --language-only \
      --enable-metrics \
    2>&1 | tail -20

  echo "Container started. First boot loads ~18 GB + drafter. Watch readiness:"
  echo "  ./run-sglang-godspeed.sh status"
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
