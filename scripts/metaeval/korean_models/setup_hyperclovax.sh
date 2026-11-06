#!/bin/bash
# =============================================================================
# HyperCLOVAX-SEED-Think-32B Setup Script
# Uses OmniServe (official deployment method from HuggingFace)
# https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
OMNISERVE_DIR="$PROJECT_ROOT/external/OmniServe"

echo "=============================================="
echo "HyperCLOVAX-SEED-Think-32B Setup"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Models dir: $MODELS_DIR"
echo "OmniServe dir: $OMNISERVE_DIR"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

echo "✅ Docker and Docker Compose available"

# Step 1: Clone OmniServe if not exists
if [ ! -d "$OMNISERVE_DIR" ]; then
    echo ""
    echo "📥 Cloning OmniServe..."
    mkdir -p "$(dirname "$OMNISERVE_DIR")"
    git clone https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe.git "$OMNISERVE_DIR"
else
    echo "✅ OmniServe already cloned"
fi

# Step 2: Download model if not exists
MODEL_PATH="$MODELS_DIR/HyperCLOVAX-SEED-Think-32B"
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "📥 Downloading HyperCLOVAX-SEED-Think-32B (~60GB)..."
    echo "   This may take a while..."
    mkdir -p "$MODELS_DIR"
    huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Think-32B \
        --local-dir "$MODEL_PATH"
else
    echo "✅ Model already downloaded at $MODEL_PATH"
fi

# Step 3: Convert model to component format
TRACK_A_DIR="$OMNISERVE_DIR/track_a"
if [ ! -d "$TRACK_A_DIR/llm" ]; then
    echo ""
    echo "🔄 Converting model to component format..."
    cd "$OMNISERVE_DIR"
    pip install huggingface_hub safetensors torch openai easydict -q
    python convert_model.py \
        --input "$MODEL_PATH" \
        --output "$TRACK_A_DIR" \
        --track a
else
    echo "✅ Model already converted"
fi

# Step 4: Configure environment
ENV_FILE="$OMNISERVE_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "📝 Creating .env configuration..."
    cp "$OMNISERVE_DIR/.env.example" "$ENV_FILE"
fi

# Update .env with correct paths
cat > "$ENV_FILE" << EOF
VLM_MODEL_PATH=$TRACK_A_DIR/llm/HyperCLOVAX-SEED-Think-32B
VLM_ENCODER_VISION_MODEL_PATH=$TRACK_A_DIR/ve/HyperCLOVAX-SEED-Think-32B
EOF

echo "✅ Environment configured"
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To start the server:"
echo "  cd $OMNISERVE_DIR"
echo "  docker compose --profile track-a build"
echo "  docker compose --profile track-a up -d"
echo ""
echo "Wait ~5 minutes for model loading, then test:"
echo "  curl http://localhost:8000/a/v1/models"
echo ""
echo "API endpoint: http://localhost:8000/a/v1"
