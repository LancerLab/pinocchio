#!/usr/bin/env python3
"""Script to switch between different Pinocchio configuration modes."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def switch_mode(mode: str, config_dir: str = "configs"):
    """Switch to a specific configuration mode."""
    config_path = Path(config_dir) / f"{mode}.json"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print(f"Available modes: {list_available_modes(config_dir)}")
        return False

    # Copy the mode config to pinocchio.json
    try:
        shutil.copy2(config_path, "pinocchio.json")
        print(f"✅ Switched to {mode} mode")
        print(f"📁 Configuration copied from: {config_path}")

        # Show mode description
        show_mode_info(mode)
        return True

    except Exception as e:
        print(f"❌ Failed to switch to {mode} mode: {e}")
        return False


def list_available_modes(config_dir: str = "configs"):
    """List all available configuration modes."""
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"❌ Config directory not found: {config_path}")
        return []

    modes = []
    for file in config_path.glob("*.json"):
        mode = file.stem
        modes.append(mode)

    return modes


def show_mode_info(mode: str):
    """Show information about a specific mode."""
    mode_info = {
        "development": {
            "description": "Full verbose logging for development",
            "features": [
                "✅ All verbose logging enabled",
                "✅ Performance tracking",
                "✅ Session tracking",
                "✅ Export on exit",
                "✅ Detailed agent communications",
                "✅ LLM request/response logging",
            ],
        },
        "production": {
            "description": "Minimal logging for end users",
            "features": [
                "❌ Verbose logging disabled",
                "❌ Performance tracking disabled",
                "❌ Session tracking disabled",
                "✅ Basic progress updates only",
                "✅ Clean user experience",
            ],
        },
        "debug": {
            "description": "Maximum logging for troubleshooting",
            "features": [
                "✅ Maximum verbose logging",
                "✅ Raw prompt/response logging",
                "✅ Internal state logging",
                "✅ Memory operations logging",
                "✅ Configuration change logging",
                "✅ Maximum recursion depth (10)",
            ],
        },
    }

    if mode in mode_info:
        info = mode_info[mode]
        print(f"\n📋 {mode.upper()} MODE:")
        print(f"   {info['description']}")
        print("\n   Features:")
        for feature in info["features"]:
            print(f"   {feature}")


def show_current_mode():
    """Show the current configuration mode."""
    try:
        with open("pinocchio.json", "r") as f:
            config = json.load(f)

        verbose_config = config.get("verbose", {})
        mode = verbose_config.get("mode", "unknown")
        enabled = verbose_config.get("enabled", False)
        level = verbose_config.get("level", "unknown")

        print(f"🔧 Current Mode: {mode.upper()}")
        print(f"   Verbose Enabled: {'✅' if enabled else '❌'}")
        print(f"   Verbose Level: {level}")

        show_mode_info(mode)

    except Exception as e:
        print(f"❌ Failed to read current configuration: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Switch between Pinocchio configuration modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/switch_mode.py development    # Switch to development mode
  python scripts/switch_mode.py production     # Switch to production mode
  python scripts/switch_mode.py debug          # Switch to debug mode
  python scripts/switch_mode.py --list         # List available modes
  python scripts/switch_mode.py --current      # Show current mode
        """,
    )

    parser.add_argument(
        "mode", nargs="?", help="Mode to switch to (development/production/debug)"
    )

    parser.add_argument("--list", action="store_true", help="List available modes")

    parser.add_argument("--current", action="store_true", help="Show current mode")

    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing configuration files (default: configs)",
    )

    args = parser.parse_args()

    if args.list:
        modes = list_available_modes(args.config_dir)
        if modes:
            print("📁 Available modes:")
            for mode in modes:
                print(f"   • {mode}")
        else:
            print("❌ No configuration modes found")
        return

    if args.current:
        show_current_mode()
        return

    if not args.mode:
        parser.print_help()
        return

    # Switch to the specified mode
    success = switch_mode(args.mode, args.config_dir)
    if success:
        print("\n🚀 You can now run Pinocchio with the new configuration:")
        print("   python -m pinocchio.cli.main")
        print("   # or")
        print("   python -m pinocchio.cli.main --config configs/%s.json", args.mode)


if __name__ == "__main__":
    main()
