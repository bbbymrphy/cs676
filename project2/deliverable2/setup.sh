#!/bin/bash

echo "========================================"
echo "Time Series Analysis Setup"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set your Anthropic API key:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""
echo "2. Run the Gradio app:"
echo "   python gradio_app.py"
echo ""
echo "3. Open your browser to:"
echo "   http://localhost:7860"
echo ""
echo "========================================"
