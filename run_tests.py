"""
Test Runner Script (run_tests.py)
---------------------------------
Discovers and executes all unit tests across the test suite.
Directs test output to stdout to avoid Windows PowerShell NativeCommandError.
"""

import sys
from pathlib import Path
import unittest

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
