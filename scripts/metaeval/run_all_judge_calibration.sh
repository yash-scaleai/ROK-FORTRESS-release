#!/bin/bash
#
# Master orchestrator for judge calibration
# 
# Execution strategy:
# 1. Run all 10 API models in parallel (no GPU needed)
#
# Usage: 
#   ./run_all_judge_calibration.sh               # Run API models (tmux by default)
#   ./run_all_judge_calibration.sh --api-only    # Same as default
#   ./run_all_judge_calibration.sh --no-tmux     # Run in current shell
#   ./run_all_judge_calibration.sh --no-refresh-dataset # Skip dataset refresh
#
# NOTE:
# Korean models are handled by dedicated scripts under
# `scripts/metaeval/korean_models/`.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BASE_DIR"

INPUT="runs/judge-calibration/calibration_dataset.jsonl"
OUTPUT_DIR="runs/judge-calibration"
LOG_DIR="$OUTPUT_DIR/logs"
CONCURRENCY_API=30
CONCURRENCY_VLLM=10
TIMEOUT_API=180
TIMEOUT_VLLM=300

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Parse flags
API_ONLY=false
USE_TMUX=true
REFRESH_DATASET=true
INSIDE_TMUX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-only)
            API_ONLY=true
            shift
            ;;
        --no-tmux)
            USE_TMUX=false
            shift
            ;;
        --no-refresh-dataset)
            REFRESH_DATASET=false
            shift
            ;;
        --inside-tmux)
            INSIDE_TMUX=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--api-only | --no-tmux | --no-refresh-dataset]"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "JUDGE CALIBRATION - MASTER ORCHESTRATOR"
echo "================================================================================"
echo ""
echo "Input: $INPUT"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "Tmux: $USE_TMUX"
echo "Refresh dataset: $REFRESH_DATASET"
echo ""

# ============================================================================
# Tmux Orchestration (default)
# ============================================================================

if [[ "$USE_TMUX" == "true" && "$INSIDE_TMUX" == "false" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "❌ tmux not found. Install tmux or rerun with --no-tmux."
        exit 1
    fi

    echo "Starting tmux sessions..."
    echo ""

    tmux new-session -d -s jc-api \
        "cd $BASE_DIR && $SCRIPT_DIR/run_all_judge_calibration.sh --inside-tmux"
    echo "  ✓ API session: jc-api"

    echo ""
    echo "Attach with:"
    echo "  tmux attach -t jc-api"
    echo ""
    exit 0
fi

# ============================================================================
# Dataset Refresh
# ============================================================================

if [[ "$REFRESH_DATASET" == "true" ]]; then
    echo "Refreshing calibration dataset..."
    uv run python scripts/metaeval/prepare_judge_calibration_data.py
    echo ""
fi

# ============================================================================
# Utility Functions
# ============================================================================

cleanup_gpu() {
    echo "Cleaning up GPU processes..."
    
    # Kill by PID files first
    for pidfile in /tmp/vllm_*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            echo "  Killing vLLM server with PID $pid"
            kill -9 "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        fi
    done
    
    # Kill any remaining vLLM processes
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    pkill -9 -f "VLLM::Engine" 2>/dev/null || true
    
    sleep 3
    
    echo "GPU memory status:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

wait_for_server() {
    local port=$1
    local timeout=${2:-300}
    local start=$(date +%s)
    
    while true; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi
        
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -ge $timeout ]; then
            return 1
        fi
        sleep 5
    done
}

run_vllm_judge() {
    local MODEL_PATH=$1
    local MODEL_KEY=$2
    local TP_SIZE=$3
    local REASONING_PARSER=$4
    local EXTRA_ARGS=$5
    local PORT=9999
    
    # Ensure port is free before starting
    fuser -k $PORT/tcp 2>/dev/null || true
    sleep 2
    
    echo ""
    echo "──────────────────────────────────────────────────────────────────────────────"
    echo "Model: $MODEL_KEY"
    echo "Path: $MODEL_PATH"
    echo "Tensor Parallel: $TP_SIZE GPUs"
    echo "Reasoning Parser: ${REASONING_PARSER:-none}"
    echo "──────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # Build GPU list (use first N GPUs)
    local GPU_LIST=$(seq -s',' 0 $((TP_SIZE - 1)))
    
    # Set cache dir to avoid /tmp noexec issues
    export TORCHINDUCTOR_CACHE_DIR="$BASE_DIR/.cache/torchinductor"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    
    # Build vLLM command - use "vllm serve" like working scripts
    local VLLM_CMD="CUDA_VISIBLE_DEVICES=$GPU_LIST TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR uv run vllm serve $MODEL_PATH"
    VLLM_CMD+=" --host 0.0.0.0"
    VLLM_CMD+=" --port $PORT"
    VLLM_CMD+=" --tensor-parallel-size $TP_SIZE"
    VLLM_CMD+=" --trust-remote-code"
    VLLM_CMD+=" --max-model-len 8192"
    VLLM_CMD+=" --enforce-eager"  # Avoid Triton compilation issues
    
    if [ -n "$REASONING_PARSER" ]; then
        VLLM_CMD+=" --reasoning-parser $REASONING_PARSER"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        VLLM_CMD+=" $EXTRA_ARGS"
    fi
    
    echo "Launching vLLM server..."
    echo "Command: $VLLM_CMD"
    
    # Launch in background
    eval "$VLLM_CMD" > "$LOG_DIR/vllm_$MODEL_KEY.log" 2>&1 &
    local VLLM_PID=$!
    echo $VLLM_PID > /tmp/vllm_$MODEL_KEY.pid
    
    echo "Waiting for server to be ready (PID: $VLLM_PID)..."
    
    if wait_for_server $PORT 600; then
        echo "✓ vLLM server ready on port $PORT"
        
        # Run judge calibration
        echo "Running judge calibration..."
        uv run python scripts/metaeval/run_judge_calibration.py \
            --judge-model "vllm/$MODEL_PATH" \
            --input "$INPUT" \
            --output-dir "$OUTPUT_DIR" \
            --concurrency $CONCURRENCY_VLLM \
            --timeout $TIMEOUT_VLLM \
            --vllm-endpoints "http://localhost:$PORT/v1" \
            2>&1 | tee "$LOG_DIR/judge_$MODEL_KEY.log"
        
        echo "✓ Judge calibration completed for $MODEL_KEY"
    else
        echo "❌ vLLM server failed to start for $MODEL_KEY"
        echo "Check log: $LOG_DIR/vllm_$MODEL_KEY.log"
        tail -50 "$LOG_DIR/vllm_$MODEL_KEY.log"
    fi
    
    # Cleanup - use -9 like working scripts
    echo "Stopping vLLM server..."
    kill -9 $VLLM_PID 2>/dev/null || true
    cleanup_gpu
}

# ============================================================================
# Phase 1: API Models (Parallel Execution)
# ============================================================================

if true; then
    echo "================================================================================"
    echo "PHASE 1: API MODELS (PARALLEL EXECUTION)"
    echo "================================================================================"
    echo ""
    echo "Running 10 API models in parallel... (est. 10-15 min)"
    echo ""
    
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
    
    # Launch all API models in background
    PIDS=()
    for model in "${API_MODELS[@]}"; do
        echo "Launching: $model"
        uv run python scripts/metaeval/run_judge_calibration.py \
            --judge-model "$model" \
            --input "$INPUT" \
            --output-dir "$OUTPUT_DIR" \
            --concurrency $CONCURRENCY_API \
            --timeout $TIMEOUT_API \
            > "$LOG_DIR/api_$(echo $model | tr '/' '_').log" 2>&1 &
        
        PIDS+=($!)
    done
    
    echo ""
    echo "Waiting for all API models to complete..."
    
    # Wait for all background jobs
    FAILED=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        model=${API_MODELS[$i]}
        if wait $pid; then
            echo "✓ $model completed"
        else
            echo "⚠️  $model failed (check log: $LOG_DIR/api_$(echo $model | tr '/' '_').log)"
            FAILED=$((FAILED + 1))
        fi
    done
    
    echo ""
    echo "✅ API models completed ($((${#API_MODELS[@]} - FAILED))/${#API_MODELS[@]} succeeded)"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo "================================================================================"
echo "CALIBRATION COMPLETE"
echo "================================================================================"
echo ""

# Count results
RESULT_FILES=$(find "$OUTPUT_DIR" -name "results_*.jsonl" 2>/dev/null | wc -l)
echo "Total result files: $RESULT_FILES"
echo ""

# List results
echo "Result files:"
find "$OUTPUT_DIR" -name "results_*.jsonl" -exec basename {} \; 2>/dev/null | sort
echo ""

echo "Next step: Run calibration analysis"
echo "  uv run python scripts/metaeval/analyze_judge_calibration.py"
echo ""

echo "Logs available in: $LOG_DIR"
echo ""
