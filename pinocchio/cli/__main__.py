"""Pinocchio CLI Entry Point.

This module provides the command-line entry point for the Pinocchio CLI.
"""

import asyncio
import sys

from .main import main, run

if __name__ == "__main__":
    # Check if legacy mode is requested via command line
    if "--legacy-cli" in sys.argv:
        # Use the new main function which handles legacy mode
        asyncio.run(main())
    else:
        # Use the existing run function for backward compatibility
        run()
