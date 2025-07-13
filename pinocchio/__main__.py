#!/usr/bin/env python3
"""
Pinocchio CLI entry point.

Run with: python -m pinocchio [--legacy-cli]
"""

import asyncio

from pinocchio.cli.main import main

if __name__ == "__main__":
    asyncio.run(main())
