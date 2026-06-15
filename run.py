#!/usr/bin/env python3
"""
Quick start script for running the autonomy stack.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autonomy.main import main

if __name__ == "__main__":
    main()
