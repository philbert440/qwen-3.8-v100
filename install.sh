#!/bin/bash
# Install the pinned upstream 1Cat-vLLM release wheel into a fresh Python 3.12 venv.
#   QWEN38_VENV  venv to create (default ./.venv)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pins.env"
venv=${QWEN38_VENV:-./.venv}
base=https://github.com/1CatAI/1Cat-vLLM/releases/download/$ONECAT_TAG
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL -o "$tmp/$ONECAT_WHEEL" "$base/$ONECAT_WHEEL"
curl -fL -o "$tmp/SHA256SUMS" "$base/SHA256SUMS"
(cd "$tmp" && sha256sum --ignore-missing -c SHA256SUMS)
echo "$ONECAT_WHEEL_SHA256  $tmp/$ONECAT_WHEEL" | sha256sum -c -
uv venv --python 3.12 "$venv"
uv pip install --python "$venv/bin/python" "$tmp/$ONECAT_WHEEL" ninja
"$venv/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
