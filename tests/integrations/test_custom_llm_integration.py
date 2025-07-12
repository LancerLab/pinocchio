#!/usr/bin/env python3
"""Test script for Custom LLM integration."""

import asyncio
import json
import logging

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

pytestmark = pytest.mark.asyncio

# Mark tests that require real LLM connection to be skipped in CI
pytestmark = [pytest.mark.asyncio, pytest.mark.real_llm]

from pinocchio.config import ConfigManager
from pinocchio.config.models import LLMConfigEntry, LLMProvider
from pinocchio.llm import CustomLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


async def test_custom_llm_connection():
    """Test Custom LLM client connection."""
    console.print(Panel.fit("🔗 Testing Custom LLM Connection", style="blue"))

    # Load configuration using Pydantic models
    config_manager = ConfigManager()
    llm_config = config_manager.get_llm_config()

    console.print(f"📋 Using configuration: {llm_config.model_dump_json(indent=2)}")

    # Create Custom LLM client with Pydantic config
    client = CustomLLMClient(config=llm_config)

    try:
        # Test simple completion
        console.print("📡 Testing basic completion...")
        prompt = "Hello, can you help me with code generation?"

        response = await client.complete(prompt, agent_type="generator")
        console.print("✅ Basic completion successful!")

        # Parse and display response
        try:
            response_data = json.loads(response)
            console.print(
                Panel(
                    json.dumps(response_data, indent=2, ensure_ascii=False),
                    title="Response",
                    style="green",
                )
            )
        except json.JSONDecodeError:
            console.print(Panel(response, title="Raw Response", style="yellow"))

        return True

    except Exception as e:
        console.print(f"❌ Connection failed: {e}", style="red")
        return False


async def test_code_generation():
    """Test code generation with Custom LLM."""
    console.print(Panel.fit("🧠 Testing Code Generation", style="blue"))

    # Load configuration using Pydantic models
    config_manager = ConfigManager()
    llm_config = config_manager.get_llm_config()

    client = CustomLLMClient(config=llm_config)

    try:
        # Test code generation
        prompt = """Please generate a Choreo DSL operator for matrix multiplication.
        The operator should be optimized for performance and include proper error handling."""

        console.print("📝 Generating matrix multiplication operator...")
        response = await client.complete(prompt, agent_type="generator")

        # Parse response
        response_data = json.loads(response)

        if response_data.get("success"):
            output = response_data.get("output", {})
            code = output.get("code", "No code generated")

            # Display code with syntax highlighting
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Generated Code", style="green"))

            # Display explanation
            explanation = response_data.get("explanation", "No explanation provided")
            console.print(Panel(explanation, title="Explanation", style="blue"))

        else:
            console.print("❌ Code generation failed", style="red")
            console.print(response_data)

        return True

    except Exception as e:
        console.print(f"❌ Code generation test failed: {e}", style="red")
        return False


async def test_debugging():
    """Test debugging with Custom LLM."""
    console.print(Panel.fit("🐛 Testing Debugging", style="blue"))

    # Load configuration using Pydantic models
    config_manager = ConfigManager()
    llm_config = config_manager.get_llm_config()

    client = CustomLLMClient(config=llm_config)

    try:
        # Test debugging
        buggy_code = """
func matmul_buggy(input: tensor, output: tensor) {
    // Buggy implementation with potential issues
    for i in range(input.shape[0]) {
        for j in range(input.shape[1]) {
            output[i][j] = input[i][j] * 2;  // Missing bounds check
        }
    }
}
"""

        prompt = f"""Please analyze and debug the following Choreo DSL code.
        Identify potential issues and provide fixes:

        {buggy_code}"""

        console.print("🔍 Analyzing buggy code...")
        response = await client.complete(prompt, agent_type="debugger")

        # Parse response
        response_data = json.loads(response)

        if response_data.get("success"):
            output = response_data.get("output", {})

            # Display issues found
            issues = output.get("issues_found", [])
            if issues:
                console.print(
                    Panel(
                        "\n".join(f"• {issue}" for issue in issues),
                        title="Issues Found",
                        style="red",
                    )
                )

            # Display fixed code
            fixed_code = output.get("fixed_code", "No fixed code provided")
            if fixed_code != "No fixed code provided":
                syntax = Syntax(
                    fixed_code, "python", theme="monokai", line_numbers=True
                )
                console.print(Panel(syntax, title="Fixed Code", style="green"))

            # Display fixes applied
            fixes = output.get("fixes_applied", [])
            if fixes:
                console.print(
                    Panel(
                        "\n".join(f"✅ {fix}" for fix in fixes),
                        title="Fixes Applied",
                        style="green",
                    )
                )

        else:
            console.print("❌ Debugging failed", style="red")
            console.print(response_data)

        return True

    except Exception as e:
        console.print(f"❌ Debugging test failed: {e}", style="red")
        return False


async def test_configuration():
    """Test configuration management with Pydantic models."""
    console.print(Panel.fit("⚙️ Testing Configuration Management", style="blue"))

    try:
        # Test configuration loading
        config_manager = ConfigManager()

        # Display current configuration
        llm_config = config_manager.get_llm_config()
        console.print(
            Panel(
                llm_config.model_dump_json(indent=2),
                title="LLM Configuration",
                style="green",
            )
        )

        # Test configuration modification
        console.print("📝 Testing configuration modification...")
        config_manager.set("llm.base_url", "http://10.0.16.46:8001")
        config_manager.set("llm.model_name", "Qwen/Qwen3-32B")

        # Save configuration
        config_manager.save()
        console.print("✅ Configuration saved successfully")

        # Reload and verify
        config_manager.reload()
        new_llm_config = config_manager.get_llm_config()
        console.print(
            Panel(
                new_llm_config.model_dump_json(indent=2),
                title="Updated LLM Configuration",
                style="green",
            )
        )

        # Test configuration validation
        if config_manager.validate_config():
            console.print("✅ Configuration validation passed", style="green")
        else:
            console.print("❌ Configuration validation failed", style="red")

        return True

    except Exception as e:
        console.print(f"❌ Configuration test failed: {e}", style="red")
        return False


async def main():
    """Main test function."""
    console.print(
        Panel.fit("🎭 Pinocchio Custom LLM Integration Test", style="bold blue")
    )

    # Test configuration
    config_success = await test_configuration()

    if config_success:
        console.print("\n" + "=" * 50 + "\n")

        # Test connection
        connection_success = await test_custom_llm_connection()

        if connection_success:
            console.print("\n" + "=" * 50 + "\n")

            # Test code generation
            await test_code_generation()

            console.print("\n" + "=" * 50 + "\n")

            # Test debugging
            await test_debugging()

            console.print("\n" + "=" * 50)
            console.print("🎉 All tests completed!", style="bold green")
        else:
            console.print(
                "❌ Connection test failed. Please check your LLM server.", style="red"
            )
    else:
        console.print("❌ Configuration test failed.", style="red")


if __name__ == "__main__":
    asyncio.run(main())


def test_llm_priority_selection():
    # Multiple LLM configs with different priorities
    config_data = {
        "llm": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "sk-xxx",
                "priority": 10,
            },
            {
                "provider": "custom",
                "base_url": "http://10.0.16.46:8001",
                "model_name": "Qwen/Qwen3-32B",
                "priority": 1,
            },
            {
                "provider": "anthropic",
                "model_name": "claude-3",
                "api_key": "sk-yyy",
                "priority": 20,
            },
        ]
    }
    cm = ConfigManager()
    cm.config.llm = config_data["llm"]
    best = cm.get_llm_config()
    assert isinstance(best, LLMConfigEntry)
    assert best.provider == LLMProvider.CUSTOM
    assert best.model_name == "Qwen/Qwen3-32B"
    assert best.priority == 1


@pytest.mark.asyncio
def test_llm_priority_fallback():
    # No custom provider, openai preferred
    config_data = {
        "llm": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "sk-xxx",
                "priority": 2,
            },
            {
                "provider": "anthropic",
                "model_name": "claude-3",
                "api_key": "sk-yyy",
                "priority": 1,
            },
        ]
    }
    cm = ConfigManager()
    cm.config.llm = config_data["llm"]
    best = cm.get_llm_config()
    assert best.provider == LLMProvider.ANTHROPIC
    assert best.priority == 1


@pytest.mark.asyncio
def test_llm_single_entry():
    # Single LLM object compatibility
    config_data = {
        "llm": {
            "provider": "custom",
            "base_url": "http://localhost:8001",
            "model_name": "Qwen/Qwen3-32B",
            "priority": 1,
        }
    }
    cm = ConfigManager()
    cm.config.llm = config_data["llm"]
    best = cm.get_llm_config()
    assert best.provider == LLMProvider.CUSTOM
    assert best.model_name == "Qwen/Qwen3-32B"
    assert best.priority == 1


@pytest.mark.asyncio
def test_llm_priority_same_base_url():
    # Same base_url, different model_name, priority takes effect
    config_data = {
        "llm": [
            {
                "provider": "custom",
                "base_url": "http://localhost:8001",
                "model_name": "Qwen3-32B-A",
                "priority": 5,
            },
            {
                "provider": "custom",
                "base_url": "http://localhost:8001",
                "model_name": "Qwen3-32B-B",
                "priority": 1,
            },
        ]
    }
    cm = ConfigManager()
    cm.config.llm = config_data["llm"]
    best = cm.get_llm_config()
    assert best.provider == LLMProvider.CUSTOM
    assert best.model_name == "Qwen3-32B-B"
    assert best.priority == 1
