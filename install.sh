#!/bin/bash

# Pinocchio CLI Installation Script for Ubuntu
# This script installs the Pinocchio CLI tool

set -e

echo "🎭 Installing Pinocchio CLI..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or later."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.9 or later."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✅ pip3 detected"

# Install poetry if not installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ Poetry installed"
else
    echo "✅ Poetry already installed"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install prompt-toolkit click pygments rich

# Install the package in development mode
echo "🔧 Installing Pinocchio CLI..."
poetry install

# Create a simple entry point script
echo "📝 Creating entry point..."
cat > /tmp/pinocchio << 'EOF'
#!/usr/bin/env python3
import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from pinocchio.cli.main import run

if __name__ == "__main__":
    run()
EOF

# Make it executable
chmod +x /tmp/pinocchio

# Try to install globally (requires sudo)
if command -v sudo &> /dev/null; then
    echo "🔧 Installing globally (requires sudo)..."
    sudo cp /tmp/pinocchio /usr/local/bin/pinocchio
    echo "✅ Pinocchio CLI installed globally!"
    echo "🎉 You can now run 'pinocchio' from anywhere!"
else
    echo "⚠️  Sudo not available. Installing locally..."
    cp /tmp/pinocchio ~/.local/bin/pinocchio
    chmod +x ~/.local/bin/pinocchio
    echo "✅ Pinocchio CLI installed locally!"
    echo "🎉 You can now run 'pinocchio' from anywhere!"
    echo "💡 Make sure ~/.local/bin is in your PATH"
fi

# Clean up
rm /tmp/pinocchio

echo ""
echo "🎭 Pinocchio CLI Installation Complete!"
echo ""
echo "Usage:"
echo "  pinocchio                    # Start interactive CLI"
echo "  pinocchio --help            # Show help"
echo ""
echo "Commands:"
echo "  help                        # Show available commands"
echo "  history                     # Show conversation history"
echo "  clear                       # Clear the screen"
echo "  quit                        # Exit the application"
echo ""
echo "Enjoy using Pinocchio! 🎭"
