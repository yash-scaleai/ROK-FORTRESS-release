#!/bin/bash
# =============================================================================
# launch_vllm.sh - Launch vLLM servers with tensor parallelism and replication
# =============================================================================
#
# Supports two modes:
# 1. Single-GPU models: One instance per GPU (default)
# 2. Multi-GPU models: Multiple replicas with tensor parallelism
#
# Examples:
#   # Single-GPU model on all 8 GPUs (8 instances for load balancing)
#   ./scripts/launch_vllm.sh --model naver-hyperclovax/HyperCLOVAX-SEED-Think-14B
#
#   # 72B model needing 4 GPUs each, 2 replicas on 8 GPUs
#   ./scripts/launch_vllm.sh --model skt/A.X-4.0 --tp 4
#
# =============================================================================

set -e

# Default values
MODEL_PATH=""
GPUS=""  # Empty = auto-detect
BASE_PORT=8000
GPU_MEM_UTIL=0.9
DTYPE="auto"
MAX_MODEL_LEN=""
LOG_DIR="./vllm_logs"
STOP_MODE=false
WAIT_TIMEOUT=600  # 10 minutes for large models
TENSOR_PARALLEL=1  # Default: 1 GPU per instance

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --tp|--tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --gpu-mem-util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --stop)
            STOP_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --model MODEL_PATH [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --model              HuggingFace model ID or path"
            echo ""
            echo "Options:"
            echo "  --gpus               Comma-separated GPU IDs (default: all available)"
            echo "  --tp, --tensor-parallel  GPUs per instance for tensor parallelism (default: 1)"
            echo "  --base-port          Starting port (default: 8000)"
            echo "  --gpu-mem-util       GPU memory utilization 0-1 (default: 0.9)"
            echo "  --dtype              Data type: auto, float16, bfloat16 (default: auto)"
            echo "  --max-model-len      Max context length (default: model default)"
            echo "  --log-dir            Log directory (default: ./vllm_logs)"
            echo "  --timeout            Timeout per server in seconds (default: 600)"
            echo "  --stop               Stop all vLLM servers"
            echo ""
            echo "Examples:"
            echo "  # 14B model, 1 GPU each, 8 instances on 8 GPUs:"
            echo "  $0 --model naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
            echo ""
            echo "  # 72B model, 4 GPUs each (tensor parallel), 2 instances on 8 GPUs:"
            echo "  $0 --model skt/A.X-4.0 --tp 4 --max-model-len 16384"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Stop mode
if [ "$STOP_MODE" = true ]; then
    echo "Stopping all vLLM servers..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    echo "All vLLM servers stopped."
    exit 0
fi

# Validate model
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    exit 1
fi

# Auto-detect GPUs if not specified
if [ -z "$GPUS" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    GPUS=$(seq -s',' 0 $((NUM_GPUS - 1)))
    echo "Auto-detected $NUM_GPUS GPUs: $GPUS"
else
    NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
fi

# Parse GPU list into array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"

# Calculate number of instances
NUM_INSTANCES=$((NUM_GPUS / TENSOR_PARALLEL))
if [ $NUM_INSTANCES -eq 0 ]; then
    echo "Error: Not enough GPUs ($NUM_GPUS) for tensor_parallel_size=$TENSOR_PARALLEL"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Setup local libcuda symlink for CUDA linking fix
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_LIB="$PROJECT_DIR/local_lib"
if [ ! -f "$LOCAL_LIB/libcuda.so" ]; then
    mkdir -p "$LOCAL_LIB"
    if [ -f "/lib/x86_64-linux-gnu/libcuda.so.1" ]; then
        ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$LOCAL_LIB/libcuda.so"
        echo "Created libcuda.so symlink in $LOCAL_LIB"
    fi
fi

echo "=============================================="
echo "vLLM Multi-GPU Server Launcher"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Total GPUs: $NUM_GPUS"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "Number of Instances: $NUM_INSTANCES"
echo "Ports: $BASE_PORT - $((BASE_PORT + NUM_INSTANCES - 1))"
echo "Memory utilization: $GPU_MEM_UTIL"
echo "Log directory: $LOG_DIR"
echo "=============================================="
echo ""

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout=$2
    local start_time=$(date +%s)
    
    while true; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi
        
        local elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -ge $timeout ]; then
            return 1
        fi
        
        sleep 5
    done
}

# Build base command
BASE_CMD="uv run vllm serve $MODEL_PATH"
BASE_CMD+=" --gpu-memory-utilization $GPU_MEM_UTIL"
BASE_CMD+=" --dtype $DTYPE"
BASE_CMD+=" --host 0.0.0.0"
BASE_CMD+=" --trust-remote-code"

# Model-specific options
if [[ "$MODEL_PATH" == *"hyperclovax"* ]] || [[ "$MODEL_PATH" == *"HyperCLOVA"* ]]; then
    BASE_CMD+=" --enforce-eager"  # Required for HyperCLOVAX due to CUDA linking issues
    BASE_CMD+=" --reasoning-parser hcx"
    echo "Note: HyperCLOVAX - using --enforce-eager --reasoning-parser hcx"
elif [[ "$MODEL_PATH" == *"skt/A.X"* ]] || [[ "$MODEL_PATH" == *"A.X-4.0"* ]]; then
    echo "Note: A.X-4.0 detected (Qwen2.5-based, no reasoning mode)"
fi

if [ $TENSOR_PARALLEL -gt 1 ]; then
    BASE_CMD+=" --tensor-parallel-size $TENSOR_PARALLEL"
fi

if [ -n "$MAX_MODEL_LEN" ]; then
    BASE_CMD+=" --max-model-len $MAX_MODEL_LEN"
fi

# Launch instances sequentially
ENDPOINTS=""
PIDS=""
FAILED=0

for ((i=0; i<NUM_INSTANCES; i++)); do
    PORT=$((BASE_PORT + i))
    
    # Calculate GPU IDs for this instance
    START_GPU=$((i * TENSOR_PARALLEL))
    GPU_IDS=""
    for ((j=0; j<TENSOR_PARALLEL; j++)); do
        GPU_IDX=$((START_GPU + j))
        if [ -z "$GPU_IDS" ]; then
            GPU_IDS="${GPU_ARRAY[$GPU_IDX]}"
        else
            GPU_IDS="$GPU_IDS,${GPU_ARRAY[$GPU_IDX]}"
        fi
    done
    
    LOG_FILE="$LOG_DIR/instance${i}_port${PORT}.log"
    
    echo "[Instance $i] Starting on GPUs $GPU_IDS, port $PORT..."
    
    # Start server with LIBRARY_PATH for CUDA linking
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    LIBRARY_PATH="$LOCAL_LIB:$LIBRARY_PATH" \
    $BASE_CMD --port $PORT > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=" $PID"
    
    echo "[Instance $i] Waiting for server to be ready (timeout: ${WAIT_TIMEOUT}s)..."
    
    if wait_for_server $PORT $WAIT_TIMEOUT; then
        echo "[Instance $i] ✓ Server READY on port $PORT (PID: $PID)"
        
        ENDPOINT="http://localhost:$PORT/v1"
        if [ -z "$ENDPOINTS" ]; then
            ENDPOINTS="$ENDPOINT"
        else
            ENDPOINTS="$ENDPOINTS,$ENDPOINT"
        fi
    else
        echo "[Instance $i] ✗ Server FAILED on port $PORT"
        echo "    Check log: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# Save state
echo "$PIDS" > "$LOG_DIR/pids.txt"
echo "$ENDPOINTS" > "$LOG_DIR/endpoints.txt"

# Summary
echo "=============================================="
echo "LAUNCH COMPLETE"
echo "=============================================="
echo ""
echo "Instances started: $((NUM_INSTANCES - FAILED))/$NUM_INSTANCES"
echo ""

if [ -n "$ENDPOINTS" ]; then
    echo "Endpoints for --vllm-endpoints:"
    echo "  $ENDPOINTS"
    echo ""
    echo "Use with run_full_judging.py:"
    echo "  uv run python scripts/run_full_judging.py \\"
    echo "    --judge-model \"vllm/$MODEL_PATH\" \\"
    echo "    --response-dir runs/rok_fortress_responses_final \\"
    echo "    --output-dir runs/rok_fortress_judgements_final \\"
    echo "    --vllm-endpoints \"$ENDPOINTS\""
    echo ""
fi

echo "PIDs: $PIDS"
echo ""
echo "To stop all servers:"
echo "  $0 --stop"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED instance(s) failed to start. Check logs in $LOG_DIR"
    exit 1
fi
