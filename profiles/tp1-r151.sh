# 1x V100-SXM2-32GB, TP1, 131K context. 1Cat-vLLM 1.5.1 bundled-profile settings: FP8 E4M3 KV,
# 2048/8192 KV/Mamba blocks, default CUDA graphs, MTP K=2. Measured on 1.5.1 with the
# Uncensored-Aggressive W4A16-AWQ build. Memory utilization 0.92 was tested on SXM2 with a 69K prompt.
export VLLM_SM70_QUANT_BACKEND=turbomind
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROFILE_ARGS=(
  --dtype half --tensor-parallel-size 1
  --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e4m3 --max-model-len 131072
  --gpu-memory-utilization "${QWEN38_GPU_UTIL:-0.92}" --max-num-batched-tokens 8192 --max-num-seqs 3
  --block-size 2048 --mamba-block-size 8192 --enable-prefix-caching --mamba-cache-mode align
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' --seed 0
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --trust-remote-code
)
