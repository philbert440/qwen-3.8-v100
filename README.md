# qwen-3.8-v100

Launch profile for the Qwen3.8-27B V100 quants on stock [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).

This is not an engine fork and there is no wheel in it. The checkpoints run on the upstream 1Cat-vLLM release
wheel: its SM70 TurboMind dequant path serves the W4A16 AWQ builds, and `prepare_nvfp4_linear` (min capability 70)
serves NVFP4 on Volta. What this repo adds is the part you would otherwise reverse-engineer from model cards: the
env and flags a V100 setup needs, a serve script, a smoke test, all pinned to one upstream release and one HF
revision per checkpoint.

Model page: https://philtompkins.com/models/qwen3.8-27b
(uncensored builds: https://philtompkins.com/models/qwen3.8-27b-uncensored)

## Pinned base

| | |
|---|---|
| Engine | 1Cat-vLLM **1.5.1**, upstream release [`v1.5.1`](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1), commit `589a5d1e1a9ea7c42d533eed724a7d67c5f1779d` |
| Wheel | `1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl`, sha256 `757f2200ba6bd20f7e3e5c8fc68c7e05461df6d284baf676ac0c16ce19434959` |
| Runtime | Python 3.12, PyTorch 2.10 (cu128), CUDA 12.8 toolkit, `ninja` on `PATH` |
| Hardware | 2× V100-32GB (`tp2-card`) or 1× V100-SXM2-32GB (`tp1-r151`) |

Every pin is in [`pins.env`](pins.env). It also holds the HF revisions:

| Checkpoint | Revision |
|---|---|
| [philbert440/Qwen3.8-27B-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ) | `7908d42a71077a5e4dc458f273682b12dfe384a0` |
| [philbert440/Qwen3.8-27B-NVFP4](https://huggingface.co/philbert440/Qwen3.8-27B-NVFP4) | `8214a60ac546c3aa56121e9faa6681239349377a` |
| [philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ) | `a3c0c1c5a606b1ecd152e57bc5afe3ce8138696b` |
| [philbert440/Qwen3.8-27B-Uncensored-Cyber-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Cyber-W4A16-AWQ) | `d6db3e17ccf92e0082001019bf932b34beb81764` |
| [philbert440/Qwen3.8-27B-Uncensored-Aggressive-NVFP4](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Aggressive-NVFP4) | `a20d02eaf705de4848592dd6b0a90ddaad34600c` |
| [philbert440/Qwen3.8-27B-Uncensored-Cyber-NVFP4](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Cyber-NVFP4) | `09479278677e5ca7638148b2cff1a67ec966a08a` |

## Install

```bash
./install.sh                       # venv at ./.venv; QWEN38_VENV=... to change
```

This downloads the release wheel and `SHA256SUMS`, checks both against the pinned hash, creates a Python 3.12 venv
with `uv`, and installs the wheel plus `ninja`. Equivalent by hand:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.1/1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl ninja
```

## Download

```bash
./download.sh w4a16 ./Qwen3.8-27B-W4A16-AWQ     # or nvfp4, aggressive-w4a16, cyber-w4a16, aggressive-nvfp4, cyber-nvfp4
```

If you cloned a checkpoint before 2026-08-15, re-fetch `tokenizer.json` and `tokenizer_config.json`. An early
calibration run shipped a tokenizer with an active truncation block that broke large image inputs. The pinned revisions
have the fixed files.

## Launch

```bash
QWEN38_MODEL=./Qwen3.8-27B-W4A16-AWQ ./serve.sh                        # 2x V100, TP2 (default profile)
QWEN38_MODEL=./Qwen3.8-27B-NVFP4 QWEN38_FORMAT=nvfp4 ./serve.sh        # NVFP4 on the same profile
QWEN38_MODEL=./Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ QWEN38_PROFILE=tp1-r151 ./serve.sh   # one V100-SXM2
```

`serve.sh` reads `QWEN38_MODEL`, `QWEN38_PROFILE`, `QWEN38_FORMAT`, `QWEN38_VENV`, `QWEN38_NAME`, `HOST` and `PORT`.
Any extra arguments go to `vllm serve` after the profile, so they override it. Set `CUDA_VISIBLE_DEVICES` to pick cards.
Boot takes a few minutes (weight load, compile, CUDA graph capture); poll `/v1/models`.

### Profiles

**`tp2-card`** ([profiles/tp2-card.sh](profiles/tp2-card.sh)) is the model cards' SM70 serve command, verbatim:
TP2, FP8 E5M2 KV, 32K context, MTP with 2 draft tokens, utilization 0.78.

```bash
VLLM_SM70_QUANT_BACKEND=turbomind VLLM_SM70_FLASH_ATTN_V100=1 \
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 NCCL_P2P_DISABLE=1 \
vllm serve ./Qwen3.8-27B-W4A16-AWQ --dtype half --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.78 \
  --max-model-len 32768 --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic"}'
```

NVFP4 adds `VLLM_SM70_NVFP4_TURBOMIND=1` and `--compilation-config '{"cudagraph_mode":"piecewise"}'`
(`QWEN38_FORMAT=nvfp4`). The cards measured this profile on 1Cat-vLLM 1.2.2: 53-59 tok/s single stream for
W4A16 on 2× V100-32GB, depending on prompt and draft sampling. It has not been re-benchmarked on 1.5.1.

**`tp1-r151`** ([profiles/tp1-r151.sh](profiles/tp1-r151.sh)) uses the 1.5.1 bundled-profile settings on one card:
FP8 E4M3 KV, 131K context, 2048/8192 KV/Mamba blocks, default CUDA graphs, up to 3 sequences, MTP with 2 draft
tokens, tool calling and thinking on. It was measured on 1.5.1 with the Uncensored-Aggressive W4A16-AWQ build:
45.3 tok/s decode on one V100-SXM2-32GB. Utilization 0.92 was tested on SXM2 with a 69K-token prompt. Lower it with
`QWEN38_GPU_UTIL` on cards that also hold other work.

The 1.5.1 release prefers FP8 E4M3 KV. E5M2 still works, but it doesn't select the E4M3 acceleration paths.

### V100 notes

- `ninja` must be on `PATH`. The dequant kernels JIT-compile at startup.
- `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0` is required for `max_num_seqs > 1` with MTP. 1.5.0 still defaulted it on.
- `NCCL_P2P_DISABLE=1` is for V100 pairs without NVLink. Leaving P2P on there costs about 4× throughput. On an
  NVLinked pair, set `NCCL_P2P_DISABLE=0`.

## Smoke test

```bash
./smoke.py http://127.0.0.1:8000
```

It checks `/v1/models`, a short arithmetic answer with reasoning split out of the content, and a tool call when the
server has a tool parser (`tp1-r151` does; `tp2-card` doesn't).

## Modern GPUs

On Ampere or newer, use stock vLLM. compressed-tensors W4A16 is supported natively:

```bash
vllm serve philbert440/Qwen3.8-27B-W4A16-AWQ --max-model-len 32768 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

## Credits

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM): the SM70 engine and release wheel this profile runs on,
  built on [vLLM](https://github.com/vllm-project/vllm).
- The Qwen team for Qwen3.8-27B.

## License

Apache-2.0 (see [LICENSE](LICENSE)), matching 1Cat-vLLM and vLLM.
