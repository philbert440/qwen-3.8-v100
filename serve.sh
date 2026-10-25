#!/bin/bash
# Serve a Qwen3.8-27B V100 checkpoint on stock 1Cat-vLLM with a pinned launch profile.
#   QWEN38_MODEL    local checkpoint dir (required; see download.sh)
#   QWEN38_PROFILE  tp2-card (default) | tp1-r151
#   QWEN38_FORMAT   w4a16 (default) | nvfp4
#   QWEN38_VENV     venv from install.sh (default ./.venv)
#   QWEN38_NAME     served model name (default qwen3.8-27b)
#   HOST / PORT     bind address (default 127.0.0.1:8000)
# Extra arguments are passed to `vllm serve` after the profile and override it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pins.env"
model=${QWEN38_MODEL:?set QWEN38_MODEL to the checkpoint directory}
profile=${QWEN38_PROFILE:-tp2-card}
venv=${QWEN38_VENV:-./.venv}
[ -f "$here/profiles/$profile.sh" ] || { echo "unknown profile: $profile" >&2; exit 2; }
. "$here/profiles/$profile.sh"
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export PATH="$venv/bin:$PATH"   # ninja must be on PATH: the dequant kernels JIT-compile at startup
command -v ninja >/dev/null || { echo "ninja not on PATH" >&2; exit 1; }
have=$("$venv/bin/python" -c 'import vllm; print(vllm.__version__)')
[ "$have" = "$ONECAT_VERSION" ] || echo "warning: profile pinned to 1Cat-vLLM $ONECAT_VERSION, venv has $have" >&2
exec "$venv/bin/vllm" serve "$model" \
  --served-model-name "${QWEN38_NAME:-qwen3.8-27b}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  "${PROFILE_ARGS[@]}" "$@"
