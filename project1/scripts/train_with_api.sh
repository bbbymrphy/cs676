#!/bin/bash

# Neural Network Training Script with API Key
# Sets up the environment and runs the training pipeline

echo "🧠 URL Credibility Neural Network Training"
echo "=========================================="
echo ""

# Load API key from environment or .env file
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
        echo "✅ Loaded API key from .env"
    elif [ -f ../.env ]; then
        export $(grep -v '^#' ../.env | xargs)
        echo "✅ Loaded API key from .env"
    else
        echo "❌ Error: ANTHROPIC_API_KEY not set!"
        echo "Please set it in your environment or create a .env file with:"
        echo "ANTHROPIC_API_KEY=your-key-here"
        exit 1
    fi
fi

# Navigate to project root directory (parent of scripts/)
cd "$(dirname "$0")/.."

echo "✅ API Key configured"
echo "✅ Starting training..."
echo ""

# Run the training script with any arguments passed
python3 scripts/train_model.py "$@"
