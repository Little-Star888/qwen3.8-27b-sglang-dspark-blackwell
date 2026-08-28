#!/usr/bin/env bash
# Self-check for bin/podman's arg rewrite. Run: ./bin/test-shim.sh
set -euo pipefail
cd "$(dirname "$0")"
export SHIM_DRY_RUN=1
fail=0
check() { # check <expected> <args...>
  local want="$1"; shift
  local got; got=$(./podman "$@")
  [ "$got" = "$want" ] || { echo "FAIL: $*"; echo "  want: $want"; echo "  got : $got"; fail=1; }
}

check "docker image inspect lmsysorg/sglang:qwen38-27b" image exists lmsysorg/sglang:qwen38-27b
check "docker run -d --name sglang-qwen38 --gpus all --ipc host -v /m:/model:ro img" \
  run -d --name sglang-qwen38 --replace --device nvidia.com/gpu=all --ipc host -v /m:/model:ro img
check "docker run --gpus all img" run --device=nvidia.com/gpu=all img
check "docker run --device /dev/foo img" run --device /dev/foo img
check "docker ps -a --filter name=x --format {{.Names}}" ps -a --filter name=x --format '{{.Names}}'
[ "$(./podman-compose -f docker-compose.yml up -d)" = "docker compose -f docker-compose.yml up -d" ] \
  || { echo "FAIL: podman-compose"; fail=1; }

[ $fail = 0 ] && echo "ok"
exit $fail
