#!/bin/bash

# URL Credibility Analyzer - Launch Script
# This script sets up the environment and launches the Streamlit app

echo "🔍 URL Credibility Analyzer with Claude AI"
echo "=========================================="
echo ""

# Load API key from environment or .env file
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
        echo "✅ Loaded API key from .env"
    else
        echo "❌ Error: ANTHROPIC_API_KEY not set!"
        echo "Please set it in your environment or create a .env file with:"
        echo "ANTHROPIC_API_KEY=your-key-here"
        exit 1
    fi
fi
# Navigate to project directory
cd "$(dirname "$0")"

echo "✅ API Key configured"
echo "✅ Starting Streamlit app..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the app
streamlit run app.py
