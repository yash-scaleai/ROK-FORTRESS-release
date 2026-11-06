#!/bin/bash
# =============================================================================
# Chain remaining local model collections after Kanana-2 completes.
# Monitors Kanana-2 collection, then runs K-EXAONE and Solar sequentially.
# =============================================================================

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET="data/rok_fortress_public.tsv"
OUTDIR="runs/rok_fortress_responses_final"
LOCAL_LIB="$PWD/local_lib"

echo "=============================================="
echo "LOCAL MODEL CHAIN: Kanana2 -> K-EXAONE -> Solar"
echo "=============================================="

# ========================================================================
# Step 1: Wait for Kanana-2 collection to finish
# ========================================================================
echo ""
echo "[1/3] Waiting for Kanana-2 collection to finish..."
while true; do
    # Check if collect_kanana2 process is still running
    if ! pgrep -f "collect_kanana2_responses" > /dev/null 2>&1; then
        echo "  Kanana-2 collection process finished."
        break
    fi
    LOG_LINES=$(wc -l < "$OUTDIR/logs/kanana2_collection.log" 2>/dev/null || echo 0)
    echo "  $(date +%H:%M:%S) - Kanana-2 still running ($LOG_LINES log entries)"
    sleep 30
done

# Check kanana2 result
KAN_FILE="$OUTDIR/responses_kanana2_with_reasoning.jsonl"
# Kanana2 script writes to responses_kanana2.jsonl, merge if needed
if [ -f "$OUTDIR/responses_kanana2.jsonl" ]; then
    echo "  Merging responses_kanana2.jsonl -> responses_kanana2_with_reasoning.jsonl"
    cat "$OUTDIR/responses_kanana2.jsonl" >> "$KAN_FILE"
    rm "$OUTDIR/responses_kanana2.jsonl"
fi
KAN_COUNT=$(wc -l < "$KAN_FILE" 2>/dev/null || echo 0)
echo "  Kanana-2: $KAN_COUNT records (target: 3810)"

# Kill Kanana-2 vLLM server
echo "  Stopping Kanana-2 vLLM..."
pkill -f "kanana-2-30b" 2>/dev/null || true
sleep 15
echo "  GPU memory after cleanup:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ========================================================================
# Step 2: K-EXAONE-236B-A23B (vLLM, TP=4, GPUs 1-4)
# ========================================================================
echo ""
echo "=============================================="
echo "[2/3] K-EXAONE-236B-A23B"
echo "=============================================="

echo "  Launching vLLM on GPUs 1,2,3,4 (TP=4)..."
CUDA_VISIBLE_DEVICES=1,2,3,4 \
LIBRARY_PATH="$LOCAL_LIB:$LIBRARY_PATH" \
nohup uv run vllm serve LGAI-EXAONE/K-EXAONE-236B-A23B \
    --host 0.0.0.0 --port 8100 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 32768 \
    --reasoning-parser deepseek_v3 \
    > vllm_logs/kexaone.log 2>&1 &

KEXAONE_PID=$!
echo "  vLLM PID: $KEXAONE_PID"

# Wait for server
echo "  Waiting for K-EXAONE to load (this may take 20+ min on EFS)..."
TIMEOUT=1800
START=$(date +%s)
while true; do
    if curl -s http://localhost:8100/v1/models 2>/dev/null | grep -q "K-EXAONE"; then
        echo "  K-EXAONE READY!"
        break
    fi
    ELAPSED=$(( $(date +%s) - START ))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "  TIMEOUT waiting for K-EXAONE after ${TIMEOUT}s"
        tail -5 vllm_logs/kexaone.log
        exit 1
    fi
    # Print loading progress every minute
    if [ $((ELAPSED % 60)) -lt 10 ]; then
        LAST_LOG=$(tail -1 vllm_logs/kexaone.log 2>/dev/null | grep -o "Loading.*" || echo "starting...")
        echo "  $(date +%H:%M:%S) ($((ELAPSED/60))m) - $LAST_LOG"
    fi
    sleep 10
done

echo "  Collecting K-EXAONE responses..."
uv run python scripts/collect_kexaone_responses.py \
    --input "$DATASET" \
    --outdir "$OUTDIR" \
    --base-url "http://localhost:8100/v1" \
    --resume \
    --concurrency 10 \
    --timeout 300

KEX_COUNT=$(wc -l < "$OUTDIR/responses_kexaone_with_reasoning.jsonl" 2>/dev/null || echo 0)
echo "  K-EXAONE: $KEX_COUNT records (target: 3810)"

echo "  Stopping K-EXAONE vLLM..."
kill $KEXAONE_PID 2>/dev/null || true
pkill -f "K-EXAONE" 2>/dev/null || true
sleep 15
echo "  GPU memory after cleanup:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# ========================================================================
# Step 3: Solar-Open-100B (Docker, TP=8, all GPUs)
# ========================================================================
echo ""
echo "=============================================="
echo "[3/3] Solar-Open-100B"
echo "=============================================="

echo "  Launching Solar via Docker (TP=8, port 8090)..."
docker run -d --name solar-open \
    --gpus all \
    --ipc=host \
    -p 8090:8000 \
    upstage/vllm-solar-open:latest \
    upstage/Solar-Open-100B \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser solar_open \
    --reasoning-parser solar_open \
    --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor \
    --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor \
    --tensor-parallel-size 8

echo "  Waiting for Solar to load..."
TIMEOUT=1800
START=$(date +%s)
while true; do
    if curl -s http://localhost:8090/v1/models 2>/dev/null | grep -q "Solar"; then
        echo "  Solar READY!"
        break
    fi
    ELAPSED=$(( $(date +%s) - START ))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "  TIMEOUT waiting for Solar after ${TIMEOUT}s"
        docker logs solar-open --tail 10
        exit 1
    fi
    if [ $((ELAPSED % 60)) -lt 10 ]; then
        LAST_LOG=$(docker logs solar-open --tail 1 2>/dev/null | grep -o "Loading.*" || echo "starting...")
        echo "  $(date +%H:%M:%S) ($((ELAPSED/60))m) - $LAST_LOG"
    fi
    sleep 10
done

echo "  Collecting Solar responses..."
uv run python scripts/collect_solar_responses.py \
    --input "$DATASET" \
    --outdir "$OUTDIR" \
    --base-url "http://localhost:8090/v1" \
    --resume \
    --reasoning \
    --concurrency 20 \
    --timeout 300

SOL_COUNT=$(wc -l < "$OUTDIR/responses_solar_open_with_reasoning.jsonl" 2>/dev/null || echo 0)
echo "  Solar: $SOL_COUNT records (target: 3810)"

echo "  Stopping Solar Docker..."
docker stop solar-open 2>/dev/null || true
docker rm solar-open 2>/dev/null || true
sleep 10

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "=============================================="
echo "ALL LOCAL MODELS COMPLETE"
echo "=============================================="
wc -l "$OUTDIR"/responses_hyperclovax_with_reasoning.jsonl \
      "$OUTDIR"/responses_kanana2_with_reasoning.jsonl \
      "$OUTDIR"/responses_kexaone_with_reasoning.jsonl \
      "$OUTDIR"/responses_solar_open_with_reasoning.jsonl
echo ""
echo "Target: 3810 records each"
