"""
Test configuration for Pinocchio project.
This file sets up pytest fixtures and environment for testing.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path to ensure correct imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixtures here if needed 