#!/bin/bash
# Download a checkpoint at the revision pinned in pins.env.
#   usage: download.sh <variant> <local-dir>
#   variant: w4a16 | nvfp4 | aggressive-w4a16 | cyber-w4a16 | aggressive-nvfp4 | cyber-nvfp4
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pins.env"
variant=${1:?variant}
dir=${2:?local dir}
case "$variant" in
  w4a16)            repo=Qwen3.8-27B-W4A16-AWQ;                        rev=$HF_REV_W4A16 ;;
  nvfp4)            repo=Qwen3.8-27B-NVFP4;                            rev=$HF_REV_NVFP4 ;;
  aggressive-w4a16) repo=Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ;  rev=$HF_REV_AGGRESSIVE_W4A16 ;;
  cyber-w4a16)      repo=Qwen3.8-27B-Uncensored-Cyber-W4A16-AWQ;       rev=$HF_REV_CYBER_W4A16 ;;
  aggressive-nvfp4) repo=Qwen3.8-27B-Uncensored-Aggressive-NVFP4;      rev=$HF_REV_AGGRESSIVE_NVFP4 ;;
  cyber-nvfp4)      repo=Qwen3.8-27B-Uncensored-Cyber-NVFP4;           rev=$HF_REV_CYBER_NVFP4 ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
hf download "philbert440/$repo" --revision "$rev" --local-dir "$dir"
