#!/bin/bash
# Run harm classifier on all models
# Usage: ./run_all_harm_classifiers.sh

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$BASE_DIR/scripts/metaeval/run_harm_classifier.py"
OUTPUT_DIR="$BASE_DIR/runs/harm-classifier-calibration"

# API Models to run in parallel
API_MODELS=(
    "openai/gpt-5.2"
    "anthropic/claude-opus-4-5"
    "anthropic/claude-sonnet-4-5"
    "gemini/gemini-3-pro-preview"
)

# vLLM Models to run sequentially
VLLM_MODELS=(
    "LGAI-EXAONE/EXAONE-4.0-32B"
    "skt/A.X-4.0"
    "naver-hyperclovax/HCX-003"
)

CONCURRENCY=30
VARIANT="A"
REASONING="medium"

echo "========================================"
echo "HARM CLASSIFIER - ALL MODELS"
echo "========================================"
echo "Concurrency: $CONCURRENCY"
echo "Variant: $VARIANT"
echo "Reasoning: $REASONING"
echo ""

# Function to run a single model
run_model() {
    local model=$1
    local log_suffix=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local log_file="$OUTPUT_DIR/log_${log_suffix}_variant_${VARIANT}.log"

    echo "[$(date '+%H:%M:%S')] Starting $model -> $log_file"

    cd "$BASE_DIR"
    uv run python "$SCRIPT" \
        --model "$model" \
        --variant "$VARIANT" \
        --reasoning-mode "$REASONING" \
        --concurrency "$CONCURRENCY" \
        --timeout 300 \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE: $model"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $model (exit $exit_code)"
    fi
    return $exit_code
}

# Run API models in parallel
echo "========== RUNNING API MODELS IN PARALLEL =========="
pids=()
for model in "${API_MODELS[@]}"; do
    run_model "$model" &
    pids+=($!)
done

# Wait for all API models to complete
echo "Waiting for ${#pids[@]} API models..."
for pid in "${pids[@]}"; do
    wait $pid || true
done

echo ""
echo "========== API MODELS COMPLETE =========="
echo ""

# Show API results
for model in "${API_MODELS[@]}"; do
    log_suffix=$(echo "$model" | tr '/' '_' | tr '-' '_')
    log_file="$OUTPUT_DIR/log_${log_suffix}_variant_${VARIANT}.log"
    if [ -f "$log_file" ]; then
        echo "=== $model ==="
        grep -E "(Results:|Exact Match Rate|Task-Level Accuracy)" "$log_file" 2>/dev/null || echo "  (no results found)"
        echo ""
    fi
done
