#!/bin/bash
# Launch HyperCLOVAX-SEED-Think-14B model with vLLM OpenAI-compatible server
# Note: This model does NOT support tensor parallelism, runs on single GPU
# 14B in bfloat16 = ~28GB, fits on a single H100 80GB

set -e

MODEL="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
PORT="${VLLM_PORT:-8000}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"  # Single GPU only
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"  # Reasonable context for judge tasks

echo "=================================================="
echo "Launching HyperCLOVAX-SEED-Think-14B vLLM Server"
echo "=================================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU: $GPU_ID"
echo "Max Model Length: $MAX_MODEL_LEN"
echo ""
echo "Note: Tensor parallelism NOT supported by this model"
echo "=================================================="

# Launch vLLM server on single GPU
# --enforce-eager disables torch.compile/CUDA graphs to avoid Triton compilation issues
CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager

