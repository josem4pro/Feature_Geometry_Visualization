#!/bin/bash
# Setup script for the Interactive LLM Latent Space Explorer
# Creates a virtual environment and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=============================================="
echo "LLM Latent Space Explorer - Setup"
echo "=============================================="
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '3\.\d+')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

echo
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo
echo "To use the explorer:"
echo
echo "1. Activate the environment:"
echo "   source $VENV_DIR/bin/activate"
echo
echo "2. Run the Streamlit dashboard:"
echo "   cd $SCRIPT_DIR"
echo "   streamlit run app.py"
echo
echo "3. Or generate static visualizations:"
echo "   python generate_visualizations.py --output ./my_vis"
echo
