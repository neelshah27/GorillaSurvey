#!/bin/bash

# Invisible Feedback - Setup Script
# Run this to set up and start the demo

set -e

echo "================================================"
echo "  Invisible Feedback - Setup & Launch"
echo "================================================"
echo ""

cd "$(dirname "$0")"

# Check Python version
echo "[1/5] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi

$PYTHON --version

# Create virtual environment if it doesn't exist
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "Created virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Dependencies installed"

# Check if .env has API key
echo ""
echo "[4/5] Checking environment..."
if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY=sk-" .env; then
        echo "API key configured"
    else
        echo "WARNING: OPENAI_API_KEY not set in .env"
        echo "LLM features will use fallbacks"
    fi
else
    echo "Creating .env file..."
    cp .env.example .env
    echo "WARNING: Please add your OPENAI_API_KEY to .env"
fi

# Start the server
echo ""
echo "[5/5] Starting server..."
echo ""
echo "================================================"
echo "  Server starting at http://localhost:8000"
echo "  Dashboard at http://localhost:8000/dashboard"
echo "  API docs at http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""

$PYTHON -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
