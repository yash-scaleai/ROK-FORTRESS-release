#!/bin/bash
# =============================================================================
# run_api_judge_parallel.sh
#
# Runs API-based judges (o4-mini, kimi-k2-thinking) on all model response files
# in parallel using tmux. Each response file gets its own tmux session.
#
# Usage:
#   ./scripts/run_api_judge_parallel.sh --judge o4-mini
#   ./scripts/run_api_judge_parallel.sh --judge kimi-k2-thinking --reasoning medium
#   ./scripts/run_api_judge_parallel.sh --judge o4-mini --dry-run
#   ./scripts/run_api_judge_parallel.sh --kill o4-mini
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
RESPONSE_DIR="runs/rok_fortress_responses"
OUTPUT_DIR="runs/final-rok-fortress-judgements"
CONCURRENCY=40
MAX_RETRIES=3
TIMEOUT=180

# Parse arguments
JUDGE=""
REASONING=""
DRY_RUN=false
KILL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --judge|-j)
            JUDGE="$2"
            shift 2
            ;;
        --reasoning|-r)
            REASONING="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --kill)
            KILL_MODE=true
            JUDGE="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --judge <judge-name> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --judge, -j     Judge name: o4-mini, kimi-k2-thinking"
            echo ""
            echo "Options:"
            echo "  --reasoning, -r Reasoning mode: low, medium, high (for kimi)"
            echo "  --dry-run       Show what would run without executing"
            echo "  --kill <judge>  Kill all sessions for specified judge"
            echo "  --concurrency   Concurrent requests per session (default: 40)"
            echo ""
            echo "Examples:"
            echo "  $0 --judge o4-mini"
            echo "  $0 --judge kimi-k2-thinking --reasoning medium"
            echo "  $0 --kill o4-mini"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate judge
if [ -z "$JUDGE" ]; then
    echo "ERROR: --judge is required"
    echo "Run with --help for usage"
    exit 1
fi

# Map judge name to full model path and session prefix
case "$JUDGE" in
    o4-mini|o4mini)
        JUDGE_MODEL="openai/o4-mini"
        SESSION_PREFIX="o4-judge"
        JUDGE_SHORT="o4_mini"
        ;;
    kimi-k2-thinking|kimi|kimi-k2)
        JUDGE_MODEL="fireworks_ai/kimi-k2-thinking"
        SESSION_PREFIX="kimi-judge"
        JUDGE_SHORT="kimi_k2_thinking"
        if [ -n "$REASONING" ]; then
            JUDGE_SHORT="${JUDGE_SHORT}_reasoning_${REASONING}"
        fi
        ;;
    *)
        echo "ERROR: Unknown judge: $JUDGE"
        echo "Supported judges: o4-mini, kimi-k2-thinking"
        exit 1
        ;;
esac

# Kill mode
if [[ "$KILL_MODE" == "true" ]]; then
    echo "Killing all $SESSION_PREFIX tmux sessions..."
    tmux ls 2>/dev/null | grep "^$SESSION_PREFIX" | cut -d: -f1 | while read session; do
        echo "  Killing: $session"
        tmux kill-session -t "$session" 2>/dev/null || true
    done
    echo "Done."
    exit 0
fi

cd "$BASE_DIR"

echo "================================================================================"
echo "API JUDGE PARALLEL RUNNER"
echo "================================================================================"
echo ""
echo "Judge model: $JUDGE_MODEL"
echo "Session prefix: $SESSION_PREFIX"
if [ -n "$REASONING" ]; then
    echo "Reasoning mode: $REASONING"
fi
echo "Response dir: $RESPONSE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Concurrency: $CONCURRENCY"
echo "Max retries: $MAX_RETRIES"
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

    session_name="$SESSION_PREFIX-$short_name"

    # Build command
    cmd="cd $BASE_DIR && uv run python scripts/run_full_judging.py"
    cmd+=" --judge-model '$JUDGE_MODEL'"
    cmd+=" --response-file '$response_file'"
    cmd+=" --output-dir '$OUTPUT_DIR'"
    cmd+=" --concurrency $CONCURRENCY"
    cmd+=" --max-retries $MAX_RETRIES"
    cmd+=" --timeout $TIMEOUT"

    if [ -n "$REASONING" ]; then
        cmd+=" --reasoning-mode $REASONING"
    fi

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

    # Small delay to avoid overwhelming API
    sleep 0.2
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
echo "  $0 --kill $JUDGE               # Kill all sessions"
echo ""
