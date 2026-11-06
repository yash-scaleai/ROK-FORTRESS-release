#!/bin/bash
# =============================================================================
# run_hyperclovax_judge_parallel.sh
#
# Runs HyperCLOVAX-SEED-Think-32B judge on all model response files in parallel.
# Uses OmniServe deployment (Docker) - NOT vanilla vLLM.
#
# Prerequisites:
#   1. OmniServe must be running: docker compose up vlm -d
#   2. GOOGLE_API_KEY must be set (for parsing model)
#
# Usage:
#   ./scripts/run_hyperclovax_judge_parallel.sh
#   ./scripts/run_hyperclovax_judge_parallel.sh --dry-run
#   ./scripts/run_hyperclovax_judge_parallel.sh --kill
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration - OmniServe uses port 10021 by default
RESPONSE_DIR="runs/rok_fortress_responses"
OUTPUT_DIR="runs/final-rok-fortress-judgements"
OMNISERVE_BASE_URL="http://localhost:10021/v1"
CONCURRENCY=25  # 20-30 tested working with OmniServe
TIMEOUT=180

# Parse arguments
DRY_RUN=false
KILL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --kill)
            KILL_MODE=true
            shift
            ;;
        --base-url)
            OMNISERVE_BASE_URL="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Show what would run without executing"
            echo "  --kill         Kill all hcx-judge tmux sessions"
            echo "  --base-url     OmniServe base URL (default: http://localhost:10021/v1)"
            echo "  --concurrency  Concurrent requests (default: 25)"
            echo ""
            echo "Prerequisites:"
            echo "  1. Start OmniServe: cd OmniServe && docker compose up vlm -d"
            echo "  2. Wait for startup: docker logs -f vlm"
            echo "  3. Verify: curl http://localhost:10021/v1/models"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Kill mode
if [[ "$KILL_MODE" == "true" ]]; then
    echo "Killing all hcx-judge tmux sessions..."
    tmux ls 2>/dev/null | grep "^hcx-judge" | cut -d: -f1 | while read session; do
        echo "  Killing: $session"
        tmux kill-session -t "$session" 2>/dev/null || true
    done
    echo "Done."
    exit 0
fi

cd "$BASE_DIR"

echo "================================================================================"
echo "HYPERCLOVAX PARALLEL JUDGE RUNNER (OmniServe)"
echo "================================================================================"
echo ""
echo "Model: HyperCLOVAX-SEED-Think-32B via OmniServe"
echo "Endpoint: $OMNISERVE_BASE_URL"
echo "Response dir: $RESPONSE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Concurrency: $CONCURRENCY"
echo ""

# Check OmniServe is running
echo "Checking OmniServe endpoint..."
if ! curl -s "$OMNISERVE_BASE_URL/models" > /dev/null 2>&1; then
    echo ""
    echo "ERROR: OmniServe not responding at $OMNISERVE_BASE_URL"
    echo ""
    echo "Please start OmniServe first:"
    echo "  cd /path/to/OmniServe"
    echo "  docker compose up vlm -d"
    echo "  docker logs -f vlm  # wait for 'Application startup complete'"
    echo ""
    exit 1
fi
echo "  OmniServe OK"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get list of response files
RESPONSE_FILES=($(ls "$RESPONSE_DIR"/responses_*.jsonl 2>/dev/null))

if [ ${#RESPONSE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No response files found in $RESPONSE_DIR"
    exit 1
fi

echo "Found ${#RESPONSE_FILES[@]} response files to process"
echo ""

# Generate session names and commands
declare -A SESSIONS

for response_file in "${RESPONSE_FILES[@]}"; do
    # Extract short name: responses_gpt-5.2_no_reasoning.jsonl -> gpt52-noreason
    filename=$(basename "$response_file" .jsonl)
    short_name=$(echo "$filename" | sed 's/responses_//' | sed 's/_with_reasoning/-withreason/' | sed 's/_no_reasoning/-noreason/' | sed 's/[._]/-/g')

    session_name="hcx-judge-$short_name"

    # Output file
    output_file="$OUTPUT_DIR/judgements_hyperclovax_on_$(basename "$response_file" .jsonl | sed 's/responses_//').jsonl"

    # Build command using the dedicated HyperCLOVAX judge script
    cmd="cd $BASE_DIR && uv run python scripts/metaeval/korean_models/run_hyperclovax_judge.py"
    cmd+=" --input '$response_file'"
    cmd+=" --output '$output_file'"
    cmd+=" --base-url '$OMNISERVE_BASE_URL'"
    cmd+=" --concurrency $CONCURRENCY"
    cmd+=" --timeout $TIMEOUT"
    cmd+="; echo ''; echo 'Press any key to close...'; read -n1"

    SESSIONS["$session_name"]="$cmd"
done

# Display sessions
echo "Sessions to create:"
echo "--------------------------------------------------------------------------------"
for session in "${!SESSIONS[@]}"; do
    echo "  $session"
done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would create ${#SESSIONS[@]} tmux sessions"
    echo ""
    echo "Sample command:"
    first_session=$(echo ${!SESSIONS[@]} | cut -d' ' -f1)
    echo "${SESSIONS[$first_session]}"
    exit 0
fi

# Launch sessions
echo "Launching ${#SESSIONS[@]} tmux sessions..."
echo ""

LAUNCHED=0
SKIPPED=0

for session in "${!SESSIONS[@]}"; do
    cmd="${SESSIONS[$session]}"

    # Check if session already exists
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "  [SKIP] $session (already running)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Create new session
    tmux new-session -d -s "$session" "$cmd"
    echo "  [START] $session"
    LAUNCHED=$((LAUNCHED + 1))

    # Small delay between launches
    sleep 0.5
done

echo ""
echo "================================================================================"
echo "LAUNCHED: $LAUNCHED sessions, SKIPPED: $SKIPPED sessions"
echo "================================================================================"
echo ""
echo "Manage sessions:"
echo "  tmux ls                        # List all sessions"
echo "  tmux attach -t <session>       # Attach to session"
echo "  Ctrl+B, D                      # Detach from session"
echo "  $0 --kill                      # Kill all sessions"
echo ""
