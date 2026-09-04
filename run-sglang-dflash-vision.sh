#!/usr/bin/env bash
# run-sglang-dflash-vision.sh — DFLASH2 + VISION (DEFAULT): DFlash2 drafter, vision tower LIVE.
# Same SGLang + NVFP4 (LMHead4) + z-lab DFlash2 stack as
# run-sglang-dflash.sh, but the vision tower stays ON (no --language-only) and
# VRAM is rebalanced so BF16 vision encoder + DFlash2 drafter both fit:
#   - --context-length 80000     (measured KV pool; was 150000 advertised)
#   - --mem-fraction-static 0.82 (same as the DSpark vision preset)
#   - --mm-feature-transport cpu (vision ON; avoids WSL2 CUDA-IPC quirk)
#
# DFLASH constraints carried over from the text-only DFLASH script:
#   - --mamba-radix-cache-strategy extra_buffer_lazy (valid with DFLASH in this
#     image; measured +5,863-token pool over eager — see README "Context pool")
#   - the DFlash2 draft pool is bounded by --speculative-draft-window-size
#     (default 8192), not by ctx — see README "Context pool" for why.
#
# DSpark launchers are UNTOUCHED. Own drafter knob DFLASH_DRAFTER_SUBDIR
# (default Qwen3.8-27B-DFlash2); shares CONTAINER_NAME=sglang-qwen38, so it
# replaces a running DSpark/DFLASH server (one GPU, swappable not simultaneous).
#
# Usage:  ./run-sglang-dflash-vision.sh [start|stop|logs|status]
# API:     http://localhost:${HOST_PORT}/v1   (also /anthropic)
# Weights:  ./setup.sh dflash2
set -uo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Fall back to bin/ shims (podman -> docker) when podman isn't installed.
PATH="$PATH:$DIR/bin"
# shellcheck disable=SC1091
[ -f "$DIR/.env" ] && source "$DIR/.env"

# DFlash2 needs a NEWER SGLang than the DSpark image: DFlash2DraftModel only
# exists upstream from 2026-08-19 (#35371), the quantized-lm_head selector
# from 2026-08-20 (#35496) — the DSpark image (qwen38-27b, commit c4271c3f,
# 2026-08-14) predates both and crashes at model load with this drafter.
# Own knob: SGGLANG_IMAGE (.env, DSpark) is ignored here on purpose.
DFLASH_SGLANG_IMAGE="${DFLASH_SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260830-a1fe4e30}"
CONTAINER="${CONTAINER_NAME:-sglang-qwen38}"
HOST_PORT="${HOST_PORT:-8040}"
CONTAINER_PORT="${CONTAINER_PORT:-30000}"
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen3.8-27B-NVFP4-RTX5090-LMHead4}"
DFLASH_DRAFTER_SUBDIR="${DFLASH_DRAFTER_SUBDIR:-Qwen3.8-27B-DFlash2}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-nvfp4}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
[ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ] || DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium"}'
# DFLASH vision knobs (overridable in .env):
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-8}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-80000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-8}"
DRAFT_WINDOW_SIZE="${DRAFT_WINDOW_SIZE:-8192}"

MODEL_HOST="$DIR/models/$MODEL_SUBDIR"
DRAFTER_HOST="$DIR/models/$DFLASH_DRAFTER_SUBDIR"

require_model() {
  [ -f "$MODEL_HOST/config.json" ] || {
    echo "ERROR: target model not at $MODEL_HOST"; echo "  Run ./setup.sh weights first."; exit 1; }
  [ -f "$DRAFTER_HOST/config.json" ] || {
    echo "ERROR: DFlash2 drafter not at $DRAFTER_HOST"; echo "  Run ./setup.sh dflash2 first."; exit 1; }
}

start() {
  require_model
  podman image exists "$DFLASH_SGLANG_IMAGE" || {
    echo "ERROR: image $DFLASH_SGLANG_IMAGE missing — run: podman pull $DFLASH_SGLANG_IMAGE"; exit 1; }

  local memfree
  memfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[gpu] free MiB: ${memfree:-?}"
  if [ -n "${memfree:-}" ] && [ "$memfree" -lt 20000 ] 2>/dev/null; then
    echo "WARNING: only ${memfree} MiB free — SGLang needs ~30 GB on the 5090. Free the GPU manually first."
  fi

  echo "Starting SGLang (DFLASH2 + VISION) -> http://localhost:${HOST_PORT}/v1"
  echo "  target : $MODEL_HOST"
  echo "  drafter: $DRAFTER_HOST (block $DFLASH_BLOCK_SIZE, draft window $DRAFT_WINDOW_SIZE)"
  echo "  ctx    : $CONTEXT_LENGTH (vision ON; same budget as the DSpark vision preset)"
  podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  podman network inspect sglang_monitor >/dev/null 2>&1 || podman network create sglang_monitor >/dev/null 2>&1 || true
  podman run -d \
    --name "$CONTAINER" --replace \
    --network sglang_monitor \
    --device nvidia.com/gpu=all \
    --ipc host \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    -v "$MODEL_HOST:/model:ro" \
    -v "$DRAFTER_HOST:/model_dflash:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e MAX_JOBS=4 \
    -e NVCC_THREADS=4 \
    -e "SGLANG_API_KEY=${API_KEY}" \
    --entrypoint python3 \
    "$DFLASH_SGLANG_IMAGE" \
    -m sglang.launch_server \
      --model-path /model \
      --served-model-name "$SERVED_MODEL_NAME" \
      --trust-remote-code \
      --tp-size 1 \
      --host 0.0.0.0 --port "$CONTAINER_PORT" \
      --kv-cache-dtype fp8_e4m3 \
      --attention-backend flashinfer \
      --context-length "$CONTEXT_LENGTH" \
      --chunked-prefill-size 2048 \
      --mamba-radix-cache-strategy "$MAMBA_RADIX_CACHE_STRATEGY" \
      --mamba-ssm-dtype bfloat16 \
      --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
      --mem-fraction-static "$MEM_FRACTION_STATIC" \
      --max-running-requests 1 \
      --speculative-algorithm DFLASH \
      --speculative-draft-model-path /model_dflash \
      --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE" \
      --speculative-draft-window-size "$DRAFT_WINDOW_SIZE" \
      --speculative-draft-model-quantization unquant \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
      --tool-call-parser qwen3_coder \
      --mm-feature-transport cpu \
      --enable-metrics \
    2>&1 | tail -20

  echo "Container started. First boot loads ~18 GB + vision tower + ~2 GB DFlash2 drafter."
  echo "  If it OOMs at boot: lower DRAFT_WINDOW_SIZE / CONTEXT_LENGTH, or drop"
  echo "  MEM_FRACTION_STATIC to 0.80, then re-run (all .env-overridable)."
  echo "Watch readiness: ./run-sglang-dflash-vision.sh status"
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
  *) echo "usage: $0 [start|stop|logs|status]"; exit 2 ;;
esac
