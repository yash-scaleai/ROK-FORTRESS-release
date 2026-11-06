#!/bin/bash
# Run all API judges for meta-eval calibration IN PARALLEL
# Usage: ./run_all_judges.sh [--reasoning-only] [--no-reasoning-only]

set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Output directory
OUTPUT_DIR="runs/metaeval-judge-calibration"
INPUT_FILE="${OUTPUT_DIR}/calibration_dataset.jsonl"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOGS_DIR"

# API models to evaluate
API_MODELS=(
    "openai/gpt-5.2"
    "openai/o4-mini"
    "anthropic/claude-opus-4-5"
    "anthropic/claude-sonnet-4-5"
    "gemini/gemini-3-pro-preview"
    "fireworks_ai/llama4-maverick-instruct-basic"
    "fireworks_ai/kimi-k2-thinking"
    "bedrock/qwen.qwen3-235b-a22b-2507-v1:0"
    "fireworks_ai/deepseek-v3p2"
    "bedrock/mistral.mistral-large-3-675b-instruct"
)

# Parse args
RUN_NO_REASONING=true
RUN_WITH_REASONING=true

if [[ "$1" == "--reasoning-only" ]]; then
    RUN_NO_REASONING=false
elif [[ "$1" == "--no-reasoning-only" ]]; then
    RUN_WITH_REASONING=false
fi

echo "========================================"
echo "META-EVAL JUDGE CALIBRATION - PARALLEL"
echo "========================================"
echo "Models: ${#API_MODELS[@]}"
echo "Run no-reasoning: $RUN_NO_REASONING"
echo "Run with-reasoning: $RUN_WITH_REASONING"
echo "Output: $OUTPUT_DIR"
echo ""

# Function to run a single judge
run_judge() {
    local model=$1
    local reasoning_mode=$2
    local safe_name=$(echo "$model" | tr '/:.-' '_')

    if [[ -n "$reasoning_mode" ]]; then
        local log_file="${LOGS_DIR}/${safe_name}_reasoning_${reasoning_mode}.log"
        echo "[$(date '+%H:%M:%S')] Starting: $model (reasoning: $reasoning_mode)"
        uv run python scripts/metaeval/run_judge_calibration.py \
            --judge-model "$model" \
            --input "$INPUT_FILE" \
            --output-dir "$OUTPUT_DIR" \
            --reasoning-mode "$reasoning_mode" \
            --concurrency 20 \
            --timeout 300 > "$log_file" 2>&1
    else
        local log_file="${LOGS_DIR}/${safe_name}_no_reasoning.log"
        echo "[$(date '+%H:%M:%S')] Starting: $model (no reasoning)"
        uv run python scripts/metaeval/run_judge_calibration.py \
            --judge-model "$model" \
            --input "$INPUT_FILE" \
            --output-dir "$OUTPUT_DIR" \
            --concurrency 30 \
            --timeout 180 > "$log_file" 2>&1
    fi

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] ✓ Completed: $model (${reasoning_mode:-no reasoning})"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAILED: $model (${reasoning_mode:-no reasoning}) - see $log_file"
    fi
    return $exit_code
}

# Track background jobs
declare -a PIDS

# Run without reasoning (all models in parallel)
if $RUN_NO_REASONING; then
    echo ""
    echo "=== PASS 1: NO REASONING (all models in parallel) ==="
    for model in "${API_MODELS[@]}"; do
        run_judge "$model" "" &
        PIDS+=($!)
    done

    # Wait for all no-reasoning jobs
    echo ""
    echo "Waiting for ${#PIDS[@]} no-reasoning jobs..."
    for pid in "${PIDS[@]}"; do
        wait $pid || true
    done
    PIDS=()
    echo ""
    echo "=== PASS 1 COMPLETE ==="
fi

# Run with reasoning (all models in parallel)
if $RUN_WITH_REASONING; then
    echo ""
    echo "=== PASS 2: WITH REASONING - medium (all models in parallel) ==="
    for model in "${API_MODELS[@]}"; do
        run_judge "$model" "medium" &
        PIDS+=($!)
    done

    # Wait for all reasoning jobs
    echo ""
    echo "Waiting for ${#PIDS[@]} reasoning jobs..."
    for pid in "${PIDS[@]}"; do
        wait $pid || true
    done
    PIDS=()
    echo ""
    echo "=== PASS 2 COMPLETE ==="
fi

echo ""
echo "========================================"
echo "ALL JUDGE RUNS COMPLETE"
echo "========================================"

# Show result summary
echo ""
echo "Result files:"
ls -la "${OUTPUT_DIR}"/results_*.jsonl 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}'

echo ""
echo "Check individual logs in: $LOGS_DIR"
