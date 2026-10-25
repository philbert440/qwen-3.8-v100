# 2x V100-32GB, TP2. The model cards' SM70 serve command: FP8 E5M2 KV, 32K context, MTP K=2.
# The cards' throughput numbers for this profile were measured on 1Cat-vLLM 1.2.2.
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0   # required for max_num_seqs > 1 with MTP
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}        # V100 pairs without NVLink; P2P on costs ~4x
PROFILE_ARGS=(
  --dtype half --attention-backend FLASH_ATTN_V100
  --tensor-parallel-size 2 --gpu-memory-utilization 0.78
  --max-model-len 32768 --kv-cache-dtype fp8_e5m2
  --enable-prefix-caching --reasoning-parser qwen3
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic"}'
)
# NVFP4 checkpoints: TurboMind NVFP4 path and piecewise graphs, as on the NVFP4 cards.
if [ "${QWEN38_FORMAT:-w4a16}" = nvfp4 ]; then
  export VLLM_SM70_NVFP4_TURBOMIND=1
  PROFILE_ARGS+=(--compilation-config '{"cudagraph_mode":"piecewise"}')
fi
