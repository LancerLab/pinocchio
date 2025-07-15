#!/usr/bin/env python3
"""
Simple LLM Connection Test
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_llm_connection():
    """Test LLM connection with current configuration."""
    try:
        # Load configuration directly
        with open("pinocchio.json", "r") as f:
            config = json.load(f)

        llm_config = config.get("llm", {})

        print("🔍 LLM Connection Test")
        print("=" * 50)
        print(f"Provider: {llm_config.get('provider', 'unknown')}")
        print(f"Base URL: {llm_config.get('base_url', 'unknown')}")
        print(f"Model: {llm_config.get('model_name', 'unknown')}")
        print(f"Timeout: {llm_config.get('timeout', 'default')}s")
        print()

        # Test basic connectivity
        base_url = llm_config.get("base_url")
        if not base_url:
            print("❌ No base_url configured")
            return False

        # Test HTTP connectivity
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                # Test basic endpoint
                test_url = f"{base_url}/v1/models"
                print(f"Testing connectivity to: {test_url}")

                async with session.get(test_url, timeout=10) as response:
                    if response.status == 200:
                        print("✅ HTTP connectivity successful")
                        models = await response.json()
                        print(f"Available models: {len(models.get('data', []))}")
                        return True
                    else:
                        print(f"❌ HTTP request failed: {response.status}")
                        return False

        except asyncio.TimeoutError:
            print("❌ Connection timeout")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


async def test_llm_completion():
    """Test LLM completion with simple prompt."""
    try:
        from pinocchio.config.models import LLMConfigEntry
        from pinocchio.llm.custom_llm_client import CustomLLMClient

        # Load configuration
        with open("pinocchio.json", "r") as f:
            config = json.load(f)

        llm_config = config.get("llm", {})

        # Create LLM client
        client_config = LLMConfigEntry(
            provider=llm_config.get("provider", "custom"),
            model_name=llm_config.get("model_name", "default"),
            base_url=llm_config.get("base_url"),
            api_key=llm_config.get("api_key"),
            timeout=llm_config.get("timeout", 30),
            max_retries=llm_config.get("max_retries", 3),
        )

        client = CustomLLMClient(client_config)

        # Test simple completion
        test_prompt = "Hello, this is a test message. Please respond with 'Connection successful'."
        print(f"Testing completion with prompt: {test_prompt}")

        response = await client.complete(test_prompt)

        if response:
            print("✅ LLM completion successful!")
            print(f"Response: {response[:200]}...")
            await client.close()
            return True
        else:
            print("❌ LLM completion failed - empty response")
            await client.close()
            return False

    except Exception as e:
        print(f"❌ LLM completion test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting LLM Connection Tests")
    print()

    # Test 1: Basic connectivity
    connectivity_ok = await test_llm_connection()
    print()

    # Test 2: LLM completion
    if connectivity_ok:
        completion_ok = await test_llm_completion()
    else:
        print("⚠️  Skipping completion test due to connectivity issues")
        completion_ok = False

    print()
    print("=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    print(f"Connectivity: {'✅ PASS' if connectivity_ok else '❌ FAIL'}")
    print(f"Completion: {'✅ PASS' if completion_ok else '❌ FAIL'}")

    if connectivity_ok and completion_ok:
        print("\n🎉 All tests passed! LLM is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check your LLM configuration.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
