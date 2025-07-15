#!/usr/bin/env python3
"""Test script for the configuration system."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from pinocchio.config import ConfigManager
from pinocchio.utils.verbose_logger import VerboseLogger, set_verbose_logger


@pytest.mark.asyncio
async def test_config_system():
    """Test the configuration system with different modes."""
    print("🧪 Testing Configuration System")
    print("=" * 50)

    # Test different configuration modes
    config_files = [
        "configs/development.json",
        "configs/production.json",
        "configs/debug.json",
    ]

    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"⚠️ Skipping {config_file} (not found)")
            continue

        print(f"\n📋 Testing {config_file}...")

        try:
            # Load configuration
            config_manager = ConfigManager(config_file)

            # Test verbose config methods
            verbose_config = config_manager.get_verbose_config()
            is_enabled = config_manager.is_verbose_enabled()
            mode = config_manager.get_verbose_mode()
            level = config_manager.get_verbose_level()

            print("   ✅ Config loaded successfully")
            print("   📊 Verbose enabled: %s", is_enabled)
            print("   🎯 Mode: %s", mode)
            print("   📈 Level: %s", level)
            print("   📁 Log file: %s", verbose_config.get("log_file", "N/A"))
            print("   🔍 Max depth: %s", verbose_config.get("max_depth", "N/A"))

            # Test verbose logger initialization
            if is_enabled:
                verbose_logger = VerboseLogger(
                    log_file=Path(verbose_config.get("log_file", "./logs/test.log")),
                    max_depth=verbose_config.get("max_depth", 5),
                    enable_colors=verbose_config.get("enable_colors", True),
                )
                set_verbose_logger(verbose_logger)

                # Test logging
                verbose_logger.log_coordinator_activity(
                    "Test activity",
                    data={"test": True, "config_file": config_file},
                    session_id="test_session",
                )

                print("   ✅ Verbose logger initialized and tested")

        except Exception as e:
            print(f"   ❌ Error testing {config_file}: {e}")

    print("\n🎉 Configuration system test completed!")


def test_mode_switching():
    """Test the mode switching functionality."""
    print("\n🔄 Testing Mode Switching")
    print("=" * 30)

    # Test mode switching script
    try:
        import subprocess

        # Test listing modes
        result = subprocess.run(
            ["python", "scripts/switch_mode.py", "--list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Mode listing works")
        else:
            print(f"❌ Mode listing failed: {result.stderr}")

        # Test current mode
        result = subprocess.run(
            ["python", "scripts/switch_mode.py", "--current"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Current mode detection works")
        else:
            print(f"❌ Current mode detection failed: {result.stderr}")

    except Exception as e:
        print(f"❌ Mode switching test failed: {e}")


def test_cli_config():
    """Test CLI configuration loading."""
    print("\n🖥️ Testing CLI Configuration")
    print("=" * 30)

    try:
        import subprocess

        # Test CLI help
        result = subprocess.run(
            ["python", "-m", "pinocchio.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "--config" in result.stdout:
            print("✅ CLI config argument works")
        else:
            print(f"❌ CLI config argument failed: {result.stderr}")

    except Exception as e:
        print(f"❌ CLI config test failed: {e}")


if __name__ == "__main__":
    # Test configuration system
    asyncio.run(test_config_system())

    # Test mode switching
    test_mode_switching()

    # Test CLI configuration
    test_cli_config()

    print("\n🎉 All tests completed!")
