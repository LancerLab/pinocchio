#!/usr/bin/env python3
"""
Test script to verify TaskResult JSON serialization fix.
"""

import json
import os
import tempfile

from pinocchio.data_models.task_planning import TaskResult
from pinocchio.utils.file_utils import safe_write_json


def test_task_result_serialization():
    print("Testing TaskResult JSON serialization...")

    # Create a TaskResult
    result = TaskResult(
        success=True,
        output={"test": "data"},
        error_message=None,
        execution_time_ms=100,
        metadata={"agent_type": "generator"},
    )

    print(f"TaskResult created: {result}")
    print(f"to_dict(): {result.to_dict()}")
    # Test JSON serialization
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    temp_file.close()

    try:
        # Test serialization
        success = safe_write_json({"test_result": result}, temp_file.name)
        print(f"Serialization successful: {success}")

        # Test deserialization
        with open(temp_file.name, "r") as f:
            data = json.load(f)
        print(f"Deserialized data: {data}")

        # Verify the structure
        test_result = data.get("test_result", {})
        assert test_result.get("success") == True
        assert test_result.get("output").get("test") == "data"
        assert test_result.get("execution_time_ms") == 100
        assert test_result.get("metadata", {}).get("agent_type") == "generator"
        print("✅ All tests passed! TaskResult JSON serialization is working.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    test_task_result_serialization()
