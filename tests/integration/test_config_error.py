#!/usr/bin/env python3
"""
Test configuration error handling.
"""

import json
import os
import tempfile
from pathlib import Path


def test_config_error_handling():
    """Test that configuration errors are properly reported."""
    print("🧪 Testing Configuration Error Handling")
    print("=" * 50)

    # Test 1: Valid configuration
    print("\n1. Testing valid configuration...")
    try:
        from pinocchio.config.config_manager import ConfigManager

        cm = ConfigManager()
        llm_config = cm.get_llm_config()
        print(f"✅ Valid config loaded: {llm_config}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    # Test 2: Invalid JSON configuration
    print("\n2. Testing invalid JSON configuration...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"llm": {"provider": "custom", "invalid": "json"')
        temp_config = f.name

    try:
        cm = ConfigManager(temp_config)
        print("❌ Should have failed with invalid JSON")
        return False
    except Exception as e:
        print(f"✅ Correctly caught invalid JSON error: {type(e).__name__}: {e}")

    # Clean up
    os.unlink(temp_config)

    # Test 3: Configuration with missing required fields
    print("\n3. Testing configuration with missing required fields...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "llm": {
                    "provider": "custom",
                    # Missing base_url and model_name
                }
            },
            f,
        )
        temp_config = f.name

    try:
        cm = ConfigManager(temp_config)
        print("❌ Should have failed with missing required fields")
        return False
    except Exception as e:
        print(f"✅ Correctly caught validation error: {type(e).__name__}: {e}")

    # Clean up
    os.unlink(temp_config)

    # Test 4: Configuration with extra fields (should fail with extra_forbidden)
    print("\n4. Testing configuration with extra fields...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "llm": {
                    "provider": "custom",
                    "base_url": "http://10.0.16.46:8001",
                    "model_name": "test-model",
                },
                "unknown_field": "should_fail",
            },
            f,
        )
        temp_config = f.name

    try:
        cm = ConfigManager(temp_config)
        print("❌ Should have failed with extra fields")
        return False
    except Exception as e:
        print(f"✅ Correctly caught extra fields error: {type(e).__name__}: {e}")

    # Clean up
    os.unlink(temp_config)

    print("\n🎉 All configuration error handling tests passed!")
    return True


if __name__ == "__main__":
    success = test_config_error_handling()
    exit(0 if success else 1)
