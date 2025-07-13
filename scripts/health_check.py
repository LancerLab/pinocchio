#!/usr/bin/env python3
"""
LLM Health Check Script

Comprehensive health check for all LLM services configured in pinocchio.json.
Includes basic performance metrics (latency, throughput) and detailed diagnostics.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --output results.json
    python scripts/health_check.py --verbose
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinocchio.config.config_manager import ConfigManager
from pinocchio.config.models import LLMConfigEntry
from pinocchio.llm.custom_llm_client import CustomLLMClient


class LLMHealthChecker:
    """Comprehensive health checker for LLM services with performance metrics."""

    def __init__(self, config_path: str = "pinocchio.json"):
        """Initialize the health checker."""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def load_config(self) -> Dict:
        """Load configuration from pinocchio.json."""
        config_manager = ConfigManager(str(self.config_path))
        return config_manager.config.model_dump()

    def extract_llm_services(self) -> List[Dict]:
        """Extract all LLM service configurations from config."""
        services = []

        # Main LLM configuration
        if "llm" in self.config and self.config["llm"]:
            services.append(
                {"name": "Main LLM", "config": self.config["llm"], "source": "llm"}
            )

        # Agent-specific LLM configurations
        agent_llm_keys = [
            "llm_generator",
            "llm_optimizer",
            "llm_debugger",
            "llm_evaluator",
        ]
        for key in agent_llm_keys:
            if key in self.config and self.config[key]:
                services.append(
                    {
                        "name": f"{key.replace('llm_', '').title()} LLM",
                        "config": self.config[key],
                        "source": key,
                    }
                )

        return services

    def normalize_service_key(self, service_config: Dict) -> str:
        """Create a unique key for service to avoid duplicates."""
        # For API-based services
        if service_config.get("provider") and service_config.get("api_key"):
            return f"{service_config['provider']}:{service_config.get('model_name', 'default')}"

        # For local IP:Port services
        if service_config.get("base_url"):
            from urllib.parse import urlparse

            parsed = urlparse(service_config["base_url"])
            return (
                f"local:{parsed.netloc}:{service_config.get('model_name', 'default')}"
            )

        # For provider-only services
        if service_config.get("provider"):
            return f"{service_config['provider']}:{service_config.get('model_name', 'default')}"

        return "unknown"

    def analyze_service_config(self, service: Dict) -> Dict:
        """Analyze a single service configuration."""
        name = service["name"]
        config = service["config"]

        analysis = {
            "name": name,
            "source": service["source"],
            "config_analysis": {
                "provider": config.get("provider", "unknown"),
                "model": config.get("model_name", "unknown"),
                "timeout": config.get("timeout", "default"),
                "max_retries": config.get("max_retries", "default"),
                "base_url": config.get("base_url"),
                "has_api_key": bool(config.get("api_key")),
                "api_key_length": len(config.get("api_key", ""))
                if config.get("api_key")
                else 0,
            },
            "issues": [],
            "recommendations": [],
        }

        # Check for configuration issues
        if not config:
            analysis["issues"].append("Empty configuration")
            return analysis

        provider = config.get("provider", "")
        base_url = config.get("base_url", "")
        api_key = config.get("api_key")

        # Analyze provider type
        if provider == "custom" and base_url:
            analysis["config_analysis"]["type"] = "local"
            analysis["config_analysis"]["endpoint"] = base_url

            if not base_url.startswith(("http://", "https://")):
                analysis["issues"].append("Invalid base_url format")
                analysis["recommendations"].append(
                    "base_url should start with http:// or https://"
                )

        elif provider in ["openai", "anthropic", "google"]:
            analysis["config_analysis"]["type"] = "api"

            if not api_key:
                analysis["issues"].append(f"Missing API key for {provider}")
                analysis["recommendations"].append(
                    f"Add api_key to {service['source']} configuration"
                )
            elif len(api_key) < 10:
                analysis["issues"].append("API key appears to be too short")
                analysis["recommendations"].append("Check API key format and length")

        else:
            analysis["issues"].append(f"Unsupported provider: {provider}")
            analysis["recommendations"].append(
                "Use 'custom' for local services or supported API providers"
            )

        # Check timeout settings
        timeout = config.get("timeout")
        if timeout and timeout < 10:
            analysis["issues"].append("Timeout too short (< 10 seconds)")
            analysis["recommendations"].append(
                "Increase timeout to at least 30 seconds"
            )

        return analysis

    async def test_service_performance(self, service: Dict) -> Dict:
        """Test performance metrics of a single LLM service."""
        analysis = self.analyze_service_config(service)
        config = service["config"]
        name = service["name"]

        # Add test results
        analysis["test_results"] = {
            "status": "not_tested",
            "latency_ms": None,
            "throughput_chars_per_sec": None,
            "response_length": None,
            "error": None,
            "tested_at": datetime.now().isoformat(),
        }

        # Skip testing if there are configuration issues
        if analysis["issues"]:
            analysis["test_results"]["status"] = "skipped"
            analysis["test_results"]["error"] = "Configuration issues prevent testing"
            return analysis

        try:
            self.logger.info(f"Testing performance of {name}...")

            # Create LLMConfigEntry object
            client_config = LLMConfigEntry(
                provider=config.get("provider", "custom"),
                model_name=config.get("model_name", "default"),
                base_url=config.get("base_url"),
                api_key=config.get("api_key"),
                timeout=config.get("timeout", 30),
                max_retries=config.get("max_retries", 3),
            )

            client = CustomLLMClient(client_config)

            # Test prompts for performance measurement
            test_prompts = [
                "Respond with exactly 'OK' if you can see this message.",
                "Say 'Hello' in one word.",
                "Reply with 'Test' only.",
            ]

            total_latency = 0
            total_response_length = 0
            successful_tests = 0

            for i, prompt in enumerate(test_prompts):
                try:
                    start_time = time.time()
                    response = await client.complete(prompt)
                    end_time = time.time()

                    if response and len(response.strip()) > 0:
                        latency = (end_time - start_time) * 1000  # Convert to ms
                        response_length = len(response.strip())

                        total_latency += latency
                        total_response_length += response_length
                        successful_tests += 1

                        self.logger.debug(
                            f"  Test {i+1}: {latency:.1f}ms, {response_length} chars"
                        )
                    else:
                        self.logger.warning(f"  Test {i+1}: Empty response")

                except Exception as e:
                    self.logger.warning(f"  Test {i+1} failed: {e}")

            await client.close()

            # Calculate performance metrics
            if successful_tests > 0:
                avg_latency = total_latency / successful_tests
                avg_response_length = total_response_length / successful_tests

                # Calculate throughput (characters per second)
                if avg_latency > 0:
                    throughput = (avg_response_length * 1000) / avg_latency  # chars/sec
                else:
                    throughput = 0

                analysis["test_results"]["status"] = "success"
                analysis["test_results"]["latency_ms"] = round(avg_latency, 1)
                analysis["test_results"]["throughput_chars_per_sec"] = round(
                    throughput, 1
                )
                analysis["test_results"]["response_length"] = avg_response_length
                analysis["test_results"]["tests_run"] = successful_tests

                # Performance analysis
                if avg_latency > 10000:
                    analysis["issues"].append("High latency (> 10 seconds)")
                    analysis["recommendations"].append(
                        "Consider optimizing network or model configuration"
                    )
                elif avg_latency > 5000:
                    analysis["recommendations"].append(
                        "Latency is acceptable but could be optimized"
                    )

                if throughput < 10:
                    analysis["issues"].append("Low throughput (< 10 chars/sec)")
                    analysis["recommendations"].append(
                        "Consider using a faster model or optimizing prompts"
                    )

                self.logger.info(
                    f"✅ {name}: Performance test passed ({avg_latency:.1f}ms, {throughput:.1f} chars/sec)"
                )
            else:
                analysis["test_results"]["status"] = "error"
                analysis["test_results"]["error"] = "All performance tests failed"
                analysis["issues"].append("Service failed all performance tests")
                self.logger.error(f"❌ {name}: All performance tests failed")

        except Exception as e:
            analysis["test_results"]["status"] = "error"
            analysis["test_results"]["error"] = str(e)
            analysis["issues"].append(f"Performance test failed: {e}")
            self.logger.error(f"❌ {name}: Performance test failed - {e}")

        return analysis

    async def check_all_services(self) -> List[Dict]:
        """Check health and performance of all configured LLM services."""
        services = self.extract_llm_services()

        if not services:
            self.logger.warning("No LLM services found in configuration")
            return []

        self.logger.info(f"Found {len(services)} LLM service(s) in configuration")

        results = []
        tested_keys = set()

        for service in services:
            service_key = self.normalize_service_key(service["config"])

            # Skip if already tested
            if service_key in tested_keys:
                self.logger.info(f"Skipping duplicate service: {service['name']}")
                continue

            tested_keys.add(service_key)
            result = await self.test_service_performance(service)
            results.append(result)

        return results

    def generate_health_report(self, results: List[Dict]) -> str:
        """Generate a comprehensive health report with performance metrics."""
        report = []
        report.append("=" * 80)
        report.append("LLM Health & Performance Check Report")
        report.append("=" * 80)
        report.append(f"Configuration file: {self.config_path}")
        report.append(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        if not results:
            report.append("❌ No LLM services found in configuration")
            return "\n".join(report)

        # Service details
        for result in results:
            name = result["name"]
            config_analysis = result["config_analysis"]
            test_results = result["test_results"]
            issues = result["issues"]
            recommendations = result["recommendations"]

            # Status indicator
            if test_results["status"] == "success":
                status_icon = "✅"
            elif test_results["status"] == "skipped":
                status_icon = "⚠️"
            else:
                status_icon = "❌"

            report.append(f"{status_icon} {name}")
            report.append(f"   Source: {result['source']}")
            report.append(f"   Type: {config_analysis.get('type', 'unknown')}")

            if "endpoint" in config_analysis:
                report.append(f"   Endpoint: {config_analysis['endpoint']}")
            if "provider" in config_analysis:
                report.append(f"   Provider: {config_analysis['provider']}")

            report.append(f"   Model: {config_analysis['model']}")
            report.append(f"   Timeout: {config_analysis['timeout']}s")
            report.append(f"   Max Retries: {config_analysis['max_retries']}")

            # Performance results
            if test_results["status"] == "success":
                report.append(f"   Latency: {test_results['latency_ms']}ms")
                report.append(
                    f"   Throughput: {test_results['throughput_chars_per_sec']} chars/sec"
                )
                report.append(
                    f"   Response Length: {test_results['response_length']:.1f} chars"
                )
                report.append(f"   Tests Run: {test_results['tests_run']}/3")
            elif test_results["status"] == "error":
                report.append(f"   Error: {test_results['error']}")

            # Issues and recommendations
            if issues:
                report.append("   Issues:")
                for issue in issues:
                    report.append(f"     • {issue}")

            if recommendations:
                report.append("   Recommendations:")
                for rec in recommendations:
                    report.append(f"     • {rec}")

            report.append("")

        # Summary statistics
        total = len(results)
        successful = sum(1 for r in results if r["test_results"]["status"] == "success")
        skipped = sum(1 for r in results if r["test_results"]["status"] == "skipped")
        failed = sum(1 for r in results if r["test_results"]["status"] == "error")

        total_issues = sum(len(r["issues"]) for r in results)
        total_recommendations = sum(len(r["recommendations"]) for r in results)

        # Performance summary
        if successful > 0:
            avg_latency = (
                sum(
                    r["test_results"]["latency_ms"]
                    for r in results
                    if r["test_results"]["status"] == "success"
                )
                / successful
            )
            avg_throughput = (
                sum(
                    r["test_results"]["throughput_chars_per_sec"]
                    for r in results
                    if r["test_results"]["status"] == "success"
                )
                / successful
            )
        else:
            avg_latency = 0
            avg_throughput = 0

        report.append("=" * 80)
        report.append("SUMMARY")
        report.append("=" * 80)
        report.append(f"Total services: {total}")
        report.append(f"✅ Successful: {successful}")
        report.append(f"⚠️  Skipped: {skipped}")
        report.append(f"❌ Failed: {failed}")
        report.append(f"Total issues found: {total_issues}")
        report.append(f"Total recommendations: {total_recommendations}")

        if successful > 0:
            report.append("")
            report.append("Performance Summary:")
            report.append(f"  Average Latency: {avg_latency:.1f}ms")
            report.append(f"  Average Throughput: {avg_throughput:.1f} chars/sec")

        report.append("")

        if successful == total:
            report.append("🎉 All LLM services are healthy!")
        elif successful > 0:
            report.append("⚠️  Some LLM services are working")
        else:
            report.append("🚨 All LLM services have issues!")

        return "\n".join(report)

    def generate_json_report(self, results: List[Dict]) -> Dict:
        """Generate JSON format report for automation."""
        successful = sum(1 for r in results if r["test_results"]["status"] == "success")

        if successful > 0:
            avg_latency = (
                sum(
                    r["test_results"]["latency_ms"]
                    for r in results
                    if r["test_results"]["status"] == "success"
                )
                / successful
            )
            avg_throughput = (
                sum(
                    r["test_results"]["throughput_chars_per_sec"]
                    for r in results
                    if r["test_results"]["status"] == "success"
                )
                / successful
            )
        else:
            avg_latency = 0
            avg_throughput = 0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_file": str(self.config_path),
            "total_services": len(results),
            "successful": successful,
            "skipped": sum(
                1 for r in results if r["test_results"]["status"] == "skipped"
            ),
            "failed": sum(1 for r in results if r["test_results"]["status"] == "error"),
            "total_issues": sum(len(r["issues"]) for r in results),
            "performance": {
                "average_latency_ms": round(avg_latency, 1),
                "average_throughput_chars_per_sec": round(avg_throughput, 1),
            },
            "services": results,
        }

        return summary


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive LLM health and performance check"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--config",
        default="pinocchio.json",
        help="Path to configuration file (default: pinocchio.json)",
    )
    parser.add_argument("--output", help="Output file for JSON results")

    args = parser.parse_args()

    try:
        # Initialize checker
        checker = LLMHealthChecker(args.config)

        if args.verbose:
            print("🔧 Configuration Analysis:")
            print(f"   Config file: {checker.config_path}")
            services = checker.extract_llm_services()
            print(f"   Found {len(services)} service(s)")
            for service in services:
                print(f"   - {service['name']} ({service['source']})")
            print()

        # Check all services
        results = await checker.check_all_services()

        # Generate and display report
        report = checker.generate_health_report(results)
        print(report)

        # Save JSON results if requested
        if args.output:
            json_report = checker.generate_json_report(results)
            with open(args.output, "w") as f:
                json.dump(json_report, f, indent=2)
            print(f"\nJSON results saved to: {args.output}")

        # Exit with appropriate code
        successful = sum(1 for r in results if r["test_results"]["status"] == "success")
        total = len(results)

        if successful == 0:
            sys.exit(1)  # All failed
        elif successful < total:
            sys.exit(2)  # Partial success
        else:
            sys.exit(0)  # All success

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
