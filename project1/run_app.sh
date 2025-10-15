#!/bin/bash

# URL Credibility Analyzer - Launch Script
# This script sets up the environment and launches the Streamlit app

echo "🔍 URL Credibility Analyzer with Claude AI"
echo "=========================================="
echo ""

# Set API key from key.py
export ANTHROPIC_API_KEY='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'

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
