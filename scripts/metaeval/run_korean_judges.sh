#!/bin/bash
#
# Sequential execution of Korean vLLM judge models with GPU management
#
# Execution order (smallest to largest for efficient loading):
# 1. HyperCLOVAX-SEED-Think (TP=2, 2 GPUs)
# 2. Kanana-2 (TP=2, 2 GPUs)
# 3. Solar-Open (TP=2, 2 GPUs)
# 4. K-EXAONE (TP=4, 4 GPUs)
# 5. A.X-K1 (TP=8, 8 GPUs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BASE_DIR"

INPUT="runs/judge-calibration/calibration_dataset.jsonl"
OUTPUT_DIR="runs/judge-calibration"
CONCURRENCY=10  # Lower for local vLLM
TIMEOUT=240

# Function to launch vLLM server
launch_vllm() {
    local MODEL_PATH=$1
    local TP_SIZE=$2
    local BASE_PORT=$3
    local MODEL_KEY=$4
    
    echo "Launching vLLM servers for $MODEL_KEY (TP=$TP_SIZE)..."
    
    # Build endpoints list
    ENDPOINTS=""
    for ((i=0; i<TP_SIZE; i++)); do
        PORT=$((BASE_PORT + i))
        ENDPOINTS="$ENDPOINTS http://localhost:$PORT/v1"
    done
    
    # Launch server using existing script
    bash scripts/launch_vllm.sh \
        --model-path "$MODEL_PATH" \
        --gpus-per-replica $TP_SIZE \
        --num-replicas 1 \
        --base-port $BASE_PORT \
        --model-name "$MODEL_KEY" \
        > "$OUTPUT_DIR/vllm_log_$MODEL_KEY.txt" 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > "$OUTPUT_DIR/vllm_$MODEL_KEY.pid"
    
    # Wait for server to be ready
    echo "Waiting for vLLM server to be ready..."
    sleep 60  # Initial wait for model loading
    
    # Check health
    for ((i=0; i<30; i++)); do
        if curl -s "http://localhost:$BASE_PORT/health" > /dev/null 2>&1; then
            echo "✓ vLLM server ready"
            return 0
        fi
        sleep 5
    done
    
    echo "❌ vLLM server failed to start"
    return 1
}

# Function to cleanup vLLM processes
cleanup_vllm() {
    local MODEL_KEY=$1
    
    echo "Cleaning up vLLM for $MODEL_KEY..."
    
    # Kill by PID file
    if [[ -f "$OUTPUT_DIR/vllm_$MODEL_KEY.pid" ]]; then
        PID=$(cat "$OUTPUT_DIR/vllm_$MODEL_KEY.pid")
        kill $PID 2>/dev/null || true
        rm "$OUTPUT_DIR/vllm_$MODEL_KEY.pid"
    fi
    
    # Kill all vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
    
    # Verify GPU freed
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
    
    echo "✓ GPU cleaned up"
}

# Function to run judge on vLLM model
run_judge() {
    local MODEL_NAME=$1
    local ENDPOINTS=$2
    
    echo "Running judge calibration with $MODEL_NAME..."
    
    uv run python scripts/metaeval/run_judge_calibration.py \
        --judge-model "$MODEL_NAME" \
        --input "$INPUT" \
        --output-dir "$OUTPUT_DIR" \
        --concurrency $CONCURRENCY \
        --timeout $TIMEOUT \
        --vllm-endpoints $ENDPOINTS
    
    echo "✓ Judging completed"
}

echo "================================================================================"
echo "KOREAN vLLM JUDGES - SEQUENTIAL EXECUTION"
echo "================================================================================"
echo ""

# ============================================================================
# 1. HyperCLOVAX-SEED-Think (TP=2, 2 GPUs)
# ============================================================================
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "1/5: HyperCLOVAX-SEED-Think (TP=2, 2 GPUs)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

launch_vllm "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B" 2 8000 "hyperclovax"
run_judge "vllm/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B" "http://localhost:8000/v1 http://localhost:8001/v1"
cleanup_vllm "hyperclovax"

# ============================================================================
# 2. Kanana-2 (TP=2, 2 GPUs)
# ============================================================================
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "2/5: Kanana-2 (TP=2, 2 GPUs)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

launch_vllm "kakaocorp/kanana-2-30b-a3b-thinking" 2 8000 "kanana-2"
run_judge "vllm/kakaocorp/kanana-2-30b-a3b-thinking" "http://localhost:8000/v1 http://localhost:8001/v1"
cleanup_vllm "kanana-2"

# ============================================================================
# 3. Solar-Open (TP=2, 2 GPUs)
# ============================================================================
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "3/5: Solar-Open (TP=2, 2 GPUs)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

launch_vllm "upstage/Solar-Open-100B" 2 8000 "solar-open"
run_judge "vllm/upstage/Solar-Open-100B" "http://localhost:8000/v1 http://localhost:8001/v1"
cleanup_vllm "solar-open"

# ============================================================================
# 4. K-EXAONE (TP=4, 4 GPUs)
# ============================================================================
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "4/5: K-EXAONE (TP=4, 4 GPUs)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

launch_vllm "LGAI-EXAONE/K-EXAONE-236B-A23B" 4 8000 "k-exaone"
run_judge "vllm/LGAI-EXAONE/K-EXAONE-236B-A23B" "http://localhost:8000/v1 http://localhost:8001/v1 http://localhost:8002/v1 http://localhost:8003/v1"
cleanup_vllm "k-exaone"

# ============================================================================
# 5. A.X-K1 (TP=8, 8 GPUs)
# ============================================================================
echo ""
echo "──────────────────────────────────────────────────────────────────────────────"
echo "5/5: A.X-K1 (TP=8, 8 GPUs - uses all)"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

launch_vllm "skt/A.X-K1" 8 8000 "a.x-k1"
run_judge "vllm/skt/A.X-K1" "http://localhost:8000/v1 http://localhost:8001/v1 http://localhost:8002/v1 http://localhost:8003/v1 http://localhost:8004/v1 http://localhost:8005/v1 http://localhost:8006/v1 http://localhost:8007/v1"
cleanup_vllm "a.x-k1"

echo ""
echo "================================================================================"
echo "ALL KOREAN MODELS COMPLETED"
echo "================================================================================"
echo ""
