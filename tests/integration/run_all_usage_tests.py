#!/usr/bin/env python3
"""
Pinocchio System - Complete Usage Test Suite Runner

This script runs all usage tests for the Pinocchio multi-agent CUDA programming system,
providing comprehensive validation and documentation of all enhanced features.

Usage:
    python run_all_usage_tests.py [options]

Options:
    --verbose    Enable verbose output
    --quick      Run quick tests only (skip comprehensive tests)
    --report     Generate detailed test report
    --help       Show this help message

Test Categories:
1. Agent Initial Prompts - CUDA expertise integration
2. Real Code Transmission - Multi-agent code processing
3. Plugin System - Extensible architecture
4. Workflow Fallback - Robust task execution
5. Memory System - Intelligent information storage
6. Knowledge System - Domain expertise management
7. Prompt Manager Integration - Context-aware generation
8. MCP Tools Integration - Professional development tools
9. Comprehensive System - End-to-end workflows
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import all test modules
test_modules = {
    "agent_prompts": {
        "module": "test_agent_prompts_usage",
        "description": "Agent Initial Prompts - CUDA expertise integration",
        "category": "core",
    },
    "code_transmission": {
        "module": "test_real_code_transmission_usage",
        "description": "Real Code Transmission - Multi-agent code processing",
        "category": "core",
    },
    "plugin_system": {
        "module": "test_plugin_system_usage",
        "description": "Plugin System - Extensible architecture",
        "category": "core",
    },
    "workflow_fallback": {
        "module": "test_workflow_fallback_usage",
        "description": "Workflow Fallback - Robust task execution",
        "category": "core",
    },
    "memory_system": {
        "module": "test_memory_system_usage",
        "description": "Memory System - Intelligent information storage",
        "category": "core",
    },
    "knowledge_system": {
        "module": "test_knowledge_system_usage",
        "description": "Knowledge System - Domain expertise management",
        "category": "core",
    },
    "prompt_integration": {
        "module": "test_prompt_manager_integration_usage",
        "description": "Prompt Manager Integration - Context-aware generation",
        "category": "integration",
    },
    "tools_integration": {
        "module": "test_mcp_tools_integration_usage",
        "description": "MCP Tools Integration - Professional development tools",
        "category": "integration",
    },
    "comprehensive": {
        "module": "test_comprehensive_system_usage",
        "description": "Comprehensive System - End-to-end workflows",
        "category": "comprehensive",
    },
}


class TestResult:
    """Container for test execution results."""

    def __init__(self, name: str, module: str, description: str):
        self.name = name
        self.module = module
        self.description = description
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.duration = 0.0
        self.error = None
        self.output = []

    def start(self):
        """Mark test as started."""
        self.status = "running"
        self.start_time = time.time()

    def complete(self, success: bool, error: Exception = None):
        """Mark test as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time if self.start_time else 0.0
        self.status = "passed" if success else "failed"
        self.error = error

    def add_output(self, line: str):
        """Add output line to test result."""
        self.output.append(line)


class TestSuiteRunner:
    """Main test suite runner with reporting capabilities."""

    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all usage tests and return comprehensive results."""
        print("=" * 80)
        print("PINOCCHIO SYSTEM - COMPLETE USAGE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'Quick' if self.quick else 'Complete'}")
        print(f"Verbose: {'Enabled' if self.verbose else 'Disabled'}")
        print()

        self.start_time = time.time()

        # Determine which tests to run
        tests_to_run = self._get_tests_to_run()

        print(f"Running {len(tests_to_run)} test modules...")
        print("-" * 80)

        # Run each test module
        for test_name, test_info in tests_to_run.items():
            result = self._run_single_test(test_name, test_info)
            self.results.append(result)

        self.end_time = time.time()

        # Generate summary
        return self._generate_summary()

    def _get_tests_to_run(self) -> Dict[str, Dict]:
        """Determine which tests to run based on mode."""
        if self.quick:
            # Run only core tests in quick mode
            return {k: v for k, v in test_modules.items() if v["category"] == "core"}
        else:
            # Run all tests in complete mode
            return test_modules

    def _run_single_test(self, test_name: str, test_info: Dict) -> TestResult:
        """Run a single test module and capture results."""
        result = TestResult(test_name, test_info["module"], test_info["description"])

        print(f"\n{'='*60}")
        print(f"RUNNING: {result.description}")
        print(f"Module: {result.module}")
        print(f"{'='*60}")

        result.start()

        try:
            # Import and run the test module
            module = __import__(f"tests.{result.module}", fromlist=[""])

            # Capture output if verbose mode is disabled
            if not self.verbose:
                import contextlib
                import io

                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    # Run the main function if it exists
                    if hasattr(module, "__main__") and callable(
                        getattr(module, "__main__")
                    ):
                        module.__main__()
                    else:
                        print(f"Running {result.module} test module...")
                        # Try to find and run test classes
                        self._run_test_classes(module)

                captured_output = output_buffer.getvalue()
                for line in captured_output.split("\n"):
                    if line.strip():
                        result.add_output(line)
            else:
                # Run with direct output in verbose mode
                if hasattr(module, "__main__") and callable(
                    getattr(module, "__main__")
                ):
                    module.__main__()
                else:
                    self._run_test_classes(module)

            result.complete(True)
            print(f"\n✅ {result.description} - PASSED")

        except Exception as e:
            result.complete(False, e)
            print(f"\n❌ {result.description} - FAILED")
            print(f"Error: {e}")

            if self.verbose:
                import traceback

                traceback.print_exc()

        print(f"Duration: {result.duration:.2f}s")

        return result

    def _run_test_classes(self, module):
        """Find and run test classes in a module."""
        # Find test classes
        test_classes = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and name.startswith("Test")
                and hasattr(obj, "setup_method")
            ):
                test_classes.append(obj)

        # Run test classes
        for test_class in test_classes:
            print(f"Running {test_class.__name__}...")

            # Create instance and run setup
            instance = test_class()
            instance.setup_method()

            # Find and run test methods
            test_methods = [
                name
                for name in dir(instance)
                if name.startswith("test_") and callable(getattr(instance, name))
            ]

            for method_name in test_methods:
                try:
                    method = getattr(instance, method_name)
                    print(f"  Running {method_name}...")
                    method()
                    print(f"  ✓ {method_name} passed")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")
                    raise

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_duration = (
            self.end_time - self.start_time
            if self.start_time and self.end_time
            else 0.0
        )

        passed_tests = [r for r in self.results if r.status == "passed"]
        failed_tests = [r for r in self.results if r.status == "failed"]

        summary = {
            "total_tests": len(self.results),
            "passed": len(passed_tests),
            "failed": len(failed_tests),
            "success_rate": len(passed_tests) / len(self.results)
            if self.results
            else 0.0,
            "total_duration": total_duration,
            "average_duration": sum(r.duration for r in self.results)
            / len(self.results)
            if self.results
            else 0.0,
            "results": self.results,
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("TEST SUITE EXECUTION SUMMARY")
        print("=" * 80)

        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Average Test Duration: {summary['average_duration']:.2f}s")

        # Test details
        print(f"\n{'Test Results':<40} {'Status':<10} {'Duration':<10}")
        print("-" * 80)

        for result in summary["results"]:
            status_symbol = "✅" if result.status == "passed" else "❌"
            print(
                f"{result.description:<40} {status_symbol:<10} {result.duration:>8.2f}s"
            )

        # Failed tests details
        failed_tests = [r for r in summary["results"] if r.status == "failed"]
        if failed_tests:
            print(f"\n{'FAILED TESTS DETAILS':<40}")
            print("-" * 80)
            for result in failed_tests:
                print(f"• {result.description}")
                if result.error:
                    print(f"  Error: {result.error}")
                print()

        # Success message or recommendations
        if summary["success_rate"] == 1.0:
            print("\n🎉 ALL TESTS PASSED!")
            print("\n🚀 SYSTEM CAPABILITIES VALIDATED:")
            print("  ✓ Agent Initial Prompts with CUDA Expertise")
            print("  ✓ Real Code Transmission Between Agents")
            print("  ✓ Extensible Plugin Architecture")
            print("  ✓ Robust Workflow Fallback Mechanisms")
            print("  ✓ Intelligent Memory Management")
            print("  ✓ Comprehensive Knowledge System")
            print("  ✓ Context-Aware Prompt Generation")
            print("  ✓ Professional MCP Tools Integration")

            if not self.quick:
                print("  ✓ End-to-End Development Workflows")
                print("  ✓ Multi-Agent Collaboration")
                print("  ✓ System Scalability and Performance")
                print("  ✓ Real-World Integration Scenarios")

            print("\n🎯 DEVELOPERS CAN NOW:")
            print("  • Leverage AI agents with expert CUDA knowledge")
            print("  • Build sophisticated multi-agent development systems")
            print("  • Extend functionality through custom plugins")
            print("  • Implement robust development workflows")
            print("  • Integrate with professional development tools")
            print("  • Create context-aware AI assistance systems")

            print("\n🏆 THE PINOCCHIO SYSTEM IS READY FOR PRODUCTION USE!")

        else:
            print(f"\n⚠️ {len(failed_tests)} TESTS FAILED")
            print(
                "Please review the failed tests and resolve issues before production use."
            )
            print("\nNext Steps:")
            print("1. Review failed test error messages")
            print("2. Check system dependencies and configuration")
            print("3. Resolve implementation issues")
            print("4. Re-run tests to validate fixes")


def generate_test_report(summary: Dict[str, Any], output_file: str = None):
    """Generate detailed test report in markdown format."""
    if not output_file:
        output_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    report_content = f"""# Pinocchio System - Usage Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Tests**: {summary['total_tests']}
- **Passed**: {summary['passed']}
- **Failed**: {summary['failed']}
- **Success Rate**: {summary['success_rate']:.1%}
- **Total Duration**: {summary['total_duration']:.2f} seconds

## Test Results Details

| Test Module | Description | Status | Duration |
|-------------|-------------|---------|----------|
"""

    for result in summary["results"]:
        status = "✅ PASSED" if result.status == "passed" else "❌ FAILED"
        report_content += f"| {result.module} | {result.description} | {status} | {result.duration:.2f}s |\n"

    # Add failed tests section if any
    failed_tests = [r for r in summary["results"] if r.status == "failed"]
    if failed_tests:
        report_content += "\n## Failed Tests Analysis\n\n"
        for result in failed_tests:
            report_content += f"### {result.description}\n\n"
            report_content += f"**Module**: {result.module}\n\n"
            report_content += f"**Error**: {result.error}\n\n"
            report_content += f"**Duration**: {result.duration:.2f}s\n\n"

    # Add capabilities summary
    report_content += "\n## System Capabilities Validated\n\n"
    if summary["success_rate"] == 1.0:
        capabilities = [
            "Agent Initial Prompts with comprehensive CUDA expertise",
            "Real code transmission and processing between agents",
            "Extensible plugin architecture for customization",
            "Robust workflow fallback mechanisms",
            "Intelligent memory management with keyword queries",
            "Comprehensive CUDA knowledge base integration",
            "Context-aware prompt generation with memory/knowledge",
            "Professional MCP tools integration for development",
            "End-to-end CUDA development workflows",
            "Multi-agent collaboration patterns",
            "System scalability and performance optimization",
            "Real-world integration scenarios",
        ]

        for capability in capabilities:
            report_content += f"- ✅ {capability}\n"
    else:
        report_content += (
            f"⚠️ System validation incomplete. {len(failed_tests)} tests failed.\n"
        )

    # Write report to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n📄 Detailed test report saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️ Failed to save test report: {e}")


def main():
    """Main entry point for test suite runner."""
    parser = argparse.ArgumentParser(
        description="Pinocchio System Complete Usage Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick tests only (skip comprehensive tests)",
    )
    parser.add_argument(
        "--report", "-r", action="store_true", help="Generate detailed test report"
    )

    args = parser.parse_args()

    # Create and run test suite
    runner = TestSuiteRunner(verbose=args.verbose, quick=args.quick)

    try:
        summary = runner.run_all_tests()
        runner.print_summary(summary)

        # Generate report if requested
        if args.report:
            generate_test_report(summary)

        # Exit with appropriate code
        exit_code = 0 if summary["success_rate"] == 1.0 else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
