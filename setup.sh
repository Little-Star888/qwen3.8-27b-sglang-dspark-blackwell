#!/usr/bin/env bash
# setup.sh — pull the SGLang image and download target + drafter.
# Safe to re-run (idempotent). No user-specific paths: everything resolves
# relative to this script's directory and is overridable via .env.
#
# By default the weights land in ./models/ (self-contained). To share a central
# store across multiple stacks, set MODELS_ROOT in .env and setup.sh will
# symlink ./models/<subdir> -> $MODELS_ROOT/<subdir> instead.
#
# Usage:
#   ./setup.sh           # pull image + download model + (default BF16) drafter
#   ./setup.sh image     # just pull the image
#   ./setup.sh weights   # just download model + (default BF16) drafter
#   ./setup.sh nvfp4-drafter   # just download the optional NVFP4 drafter (opt-out)
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[ -f "$DIR/.env" ] && source "$DIR/.env"

MODEL_REPO="${MODEL_REPO:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4}"
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen3.8-27B-NVFP4-RTX5090-LMHead4}"
# Default DSpark drafter: the BF16 RadixArk build (~2.5 GB). Better accept
# length on the 5090 (~4.0 vs ~1.7 for the NVFP4 build) -> ~171 vs ~93 t/s.
DRAFTER_REPO="${DRAFTER_REPO:-RadixArk/Qwen3.8-27B-DSpark}"
DRAFTER_SUBDIR="${DRAFTER_SUBDIR:-Qwen3.8-27B-DSpark-BF16}"
# Optional NVFP4 DSpark drafter (the older gittensor-model-hub build, ~1.4 GB).
# Opt-out: ./setup.sh nvfp4-drafter, then in .env:
#   DRAFTER_SUBDIR=Qwen3.8-27B-DSpark-NVFP4
#   DRAFT_MODEL_QUANTIZATION=modelopt_fp4
DRAFTER_NVFP4_REPO="${DRAFTER_NVFP4_REPO:-gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4}"
DRAFTER_NVFP4_SUBDIR="${DRAFTER_NVFP4_SUBDIR:-Qwen3.8-27B-DSpark-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-}"
DRAFTER_REVISION="${DRAFTER_REVISION:-}"
HF_HUB_ACCESS_TOKEN="${HF_HUB_ACCESS_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
SGGLANG_IMAGE="${SGGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
# Default: store weights inside this repo (self-contained). Override to share.
MODELS_ROOT="${MODELS_ROOT:-$DIR/models}"

MODELS="$DIR/models"                 # local dir the launchers mount
mkdir -p "$MODELS" "$MODELS_ROOT"

# If MODELS_ROOT is somewhere else, expose it under ./models/ via symlinks.
# If it *is* ./models (the default), files land in place and no link is needed.
if [ "$(cd "$MODELS_ROOT" && pwd)" != "$(cd "$MODELS" && pwd)" ]; then
  DO_SYMLINK=1
else
  DO_SYMLINK=0
fi

# link <subdir>  ->  symlink ./models/<subdir> to $MODELS_ROOT/<subdir> (if missing)
link_subdir() {
  local sub="$1"
  local root="$MODELS_ROOT/$sub"
  if [ ! -e "$MODELS/$sub" ]; then
    ln -s "$root" "$MODELS/$sub"
    echo "  linked $MODELS/$sub -> $root"
  fi
}

# Use the token if present in .env or env (HF_TOKEN / HF_HUB_ACCESS_TOKEN / file).
TOKEN_ARGS=()
if [ -n "$HF_TOKEN" ]; then
  TOKEN_ARGS+=(--token "$HF_TOKEN")
elif [ -n "$HF_HUB_ACCESS_TOKEN" ]; then
  TOKEN_ARGS+=(--token "$HF_HUB_ACCESS_TOKEN")
fi
# Xet fast transfer (resumable) — driven by env var, NOT a CLI flag
# (huggingface_hub 1.21 has no --xet flag; HF_XET_HIGH_PERFORMANCE=1 enables it).
[ "${HF_XET_HIGH_PERFORMANCE:-1}" = "1" ] && export HF_XET_HIGH_PERFORMANCE=1

pull_image() {
  echo ">>> Pulling SGLang image: $SGGLANG_IMAGE"
  podman pull "$SGGLANG_IMAGE"
}

download() {
  command -v hf >/dev/null 2>&1 || {
    echo "ERROR: 'hf' CLI not found. Install: pip install 'huggingface_hub[cli]'"; exit 2; }

  echo ">>> Downloading target model: $MODEL_REPO -> $MODELS_ROOT/$MODEL_SUBDIR"
  local model_extra=()
  [ -n "$MODEL_REVISION" ] && model_extra+=(--revision "$MODEL_REVISION")
  # no trailing pipeline: under set -e/pipefail a failed download must abort.
  hf download "$MODEL_REPO" --local-dir "$MODELS_ROOT/$MODEL_SUBDIR" \
      "${TOKEN_ARGS[@]}" "${model_extra[@]}"

  echo ">>> Downloading DSpark drafter: $DRAFTER_REPO -> $MODELS_ROOT/$DRAFTER_SUBDIR"
  local drafter_extra=()
  [ -n "$DRAFTER_REVISION" ] && drafter_extra+=(--revision "$DRAFTER_REVISION")
  hf download "$DRAFTER_REPO" --local-dir "$MODELS_ROOT/$DRAFTER_SUBDIR" \
      "${TOKEN_ARGS[@]}" "${drafter_extra[@]}"

  if [ "$DO_SYMLINK" = "1" ]; then
    echo ">>> Linking into $MODELS:"
    link_subdir "$MODEL_SUBDIR"
    link_subdir "$DRAFTER_SUBDIR"
  else
    echo ">>> Weights in $MODELS (launchers mount ./models/<subdir>). No symlink needed."
  fi

  echo ">>> Done:"
  ls -l "$MODELS" 2>/dev/null || true
  du -sh "$MODELS_ROOT/$MODEL_SUBDIR" "$MODELS_ROOT/$DRAFTER_SUBDIR" 2>/dev/null || true
}

# Download ONLY the optional NVFP4 DSpark drafter (gittensor-model-hub,
# ~1.4 GB), into its own subdir. Use for the smaller / opt-out drafter:
# set DRAFTER_SUBDIR=Qwen3.8-27B-DSpark-NVFP4 + DRAFT_MODEL_QUANTIZATION=
# modelopt_fp4 in .env.
download_nvfp4_drafter() {
  command -v hf >/dev/null 2>&1 || {
    echo "ERROR: 'hf' CLI not found. Install: pip install 'huggingface_hub[cli]'"; exit 2; }

  echo ">>> Downloading NVFP4 DSpark drafter: $DRAFTER_NVFP4_REPO -> $MODELS_ROOT/$DRAFTER_NVFP4_SUBDIR"
  hf download "$DRAFTER_NVFP4_REPO" --local-dir "$MODELS_ROOT/$DRAFTER_NVFP4_SUBDIR" \
      "${TOKEN_ARGS[@]}"

  [ "$DO_SYMLINK" = "1" ] && link_subdir "$DRAFTER_NVFP4_SUBDIR"
  echo ">>> Done. To use it, set in .env:"
  echo "    DRAFTER_SUBDIR=$DRAFTER_NVFP4_SUBDIR"
  echo "    DRAFT_MODEL_QUANTIZATION=modelopt_fp4"
}

case "${1:-all}" in
  image)        pull_image ;;
  weights)      download ;;
  all)          pull_image; download ;;
  nvfp4-drafter) download_nvfp4_drafter ;;
  *) echo "usage: $0 [image|weights|all|nvfp4-drafter]"; exit 2 ;;
esac
