"""
Entry point for the AI Trading Analysis application
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.app import main

if __name__ == "__main__":
    main() 