# Pinocchio Documentation Index

This document provides a comprehensive index of all documentation available for the Pinocchio multi-agent system v1.0.0.

## 📚 Documentation Structure

### 👥 User Guides
Essential documentation for end users:
- [Setup Guide](user-guides/setup.md) - Installation and initial setup instructions
- [Configuration Guide](user-guides/configuration.md) - Detailed configuration options
- [Configuration Summary](user-guides/CONFIGURATION_SUMMARY.md) - Quick reference for configuration
- [Verbose Configuration](user-guides/VERBOSE_CONFIGURATION.md) - Advanced logging and debugging options
- [Customization Guide](user-guides/customization.md) - How to customize and extend the system

### 🛠️ Developer Guides
Technical documentation for developers:
- [Development Guide](developer-guides/development.md) - Developer setup and contribution guidelines
- [Architecture Overview](developer-guides/architecture.md) - System design and component overview
- [Testing Guide](developer-guides/testing.md) - Testing procedures and best practices

### 📖 API Reference
Technical specifications and formats:
- [Task Planning JSON Format](api-reference/TASK_PLANNING_JSON_FORMAT.md) - JSON schema for task planning

### 🎓 Tutorials
Step-by-step guides and examples:
- [CLI Workflow End-to-End Report](tutorials/cli_workflow_end2end_report.md) - Detailed workflow documentation

### 🔧 Development Resources
Additional development tools and utilities:
- [Development Utilities](development/) - Development tools and utilities
  - [Testing README](development/README_TESTING.md) - Testing framework documentation
  - [Utils Application Summary](development/UTILS_APPLICATION_SUMMARY.md) - Utility modules overview
  - [Test Performance Optimization](development/test_performance_optimization.md) - Performance testing guide
  - [Utility Module Design](development/utility_module_design.md) - Design patterns for utilities

## 📋 Additional Documentation

### Project Information
- [Changelog](CHANGELOG.md) - Version history and changes
- [Mode Feature Summary](MODE_FEATURE_SUMMARY.md) - Overview of different operational modes
- [Agent-Specific LLM Configuration](agent_specific_llm_config.md) - Configure different LLMs for different agents

## 🚀 Quick Start Paths

### For New Users
1. [Setup Guide](user-guides/setup.md)
2. [Configuration Guide](user-guides/configuration.md)
3. [CLI Workflow Tutorial](tutorials/cli_workflow_end2end_report.md)

### For Developers
1. [Development Guide](developer-guides/development.md)
2. [Architecture Overview](developer-guides/architecture.md)
3. [Testing Guide](developer-guides/testing.md)

### For System Administrators
1. [Configuration Summary](user-guides/CONFIGURATION_SUMMARY.md)
2. [Verbose Configuration](user-guides/VERBOSE_CONFIGURATION.md)
3. [Customization Guide](user-guides/customization.md)

## 🆘 Support and Troubleshooting

- **Configuration Issues**: Check [Configuration Summary](user-guides/CONFIGURATION_SUMMARY.md)
- **Debugging**: Review [Verbose Configuration](user-guides/VERBOSE_CONFIGURATION.md)
- **Testing Problems**: See [Testing Guide](developer-guides/testing.md)
- **Development Setup**: Follow [Development Guide](developer-guides/development.md)

## 📊 Documentation Status

- ✅ User Guides: Complete and organized
- ✅ Developer Guides: Complete and organized  
- ✅ API Reference: Available
- ✅ Tutorials: Available
- ✅ Project Documentation: Up to date

## 🏗️ System Architecture Overview

Pinocchio is a multi-agent system with the following key components:

### Core Agents
- **Generator Agent**: Creates initial code implementations
- **Debugger Agent**: Analyzes and fixes code issues  
- **Evaluator Agent**: Assesses code performance and quality
- **Optimizer Agent**: Improves code performance

### System Components
- **Coordinator**: Orchestrates multi-agent workflows
- **Task Planner**: Creates execution plans for user requests
- **Memory System**: Stores agent interactions and results
- **Knowledge Management**: Maintains domain-specific knowledge
- **Configuration Management**: Centralized system configuration

### Key Features
- Multi-agent collaboration for code generation
- CUDA programming specialization
- Configurable LLM providers
- Session-based workflow management
- Comprehensive logging and debugging
- Plugin system for extensibility

---

**Note**: This documentation is for Pinocchio v1.0.0. For the latest updates, check the [Changelog](CHANGELOG.md).
