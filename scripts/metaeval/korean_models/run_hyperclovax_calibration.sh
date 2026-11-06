#!/bin/bash
# =============================================================================
# HyperCLOVAX-SEED-Think-32B Judge Calibration
# Uses OmniServe Docker deployment
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OMNISERVE_DIR="$PROJECT_ROOT/external/OmniServe"
OUTPUT_DIR="$PROJECT_ROOT/runs/judge-calibration"
LOG_DIR="$OUTPUT_DIR/logs"

# OmniServe API endpoint
OMNISERVE_API="http://localhost:8000/a/v1"

echo "=============================================="
echo "HyperCLOVAX Judge Calibration"
echo "=============================================="

mkdir -p "$LOG_DIR"

# Function to check if OmniServe is running
check_omniserve() {
    curl -s "$OMNISERVE_API/models" > /dev/null 2>&1
}

# Function to start OmniServe
start_omniserve() {
    echo "🚀 Starting OmniServe..."
    cd "$OMNISERVE_DIR"
    docker compose --profile track-a up -d
    
    echo "⏳ Waiting for OmniServe to be ready (this may take 5+ minutes)..."
    local max_wait=600  # 10 minutes
    local waited=0
    while ! check_omniserve; do
        sleep 10
        waited=$((waited + 10))
        if [ $waited -ge $max_wait ]; then
            echo "❌ OmniServe failed to start within $max_wait seconds"
            docker compose logs vlm | tail -50
            exit 1
        fi
        echo "   Still waiting... ($waited seconds)"
    done
    echo "✅ OmniServe is ready!"
}

# Function to stop OmniServe
stop_omniserve() {
    echo "🛑 Stopping OmniServe..."
    cd "$OMNISERVE_DIR"
    docker compose --profile track-a down
}

# Check if OmniServe is running
if check_omniserve; then
    echo "✅ OmniServe already running"
else
    # Check if setup was done
    if [ ! -d "$OMNISERVE_DIR" ]; then
        echo "❌ OmniServe not set up. Run setup_hyperclovax.sh first."
        exit 1
    fi
    start_omniserve
fi

# Run judge calibration
echo ""
echo "📊 Running judge calibration..."
cd "$PROJECT_ROOT"

uv run python scripts/metaeval/korean_models/run_hyperclovax_judge.py \
    --input "$OUTPUT_DIR/calibration_dataset.jsonl" \
    --output "$OUTPUT_DIR/results_omniserve_hyperclovax_seed_think_32b.jsonl" \
    --base-url "$OMNISERVE_API" \
    --concurrency 5 \
    --timeout 180 \
    2>&1 | tee "$LOG_DIR/judge_hyperclovax.log"

echo ""
echo "✅ HyperCLOVAX calibration complete!"
echo "Results: $OUTPUT_DIR/results_omniserve_hyperclovax_seed_think_32b.jsonl"

# Ask about stopping
read -p "Stop OmniServe? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    stop_omniserve
fi
