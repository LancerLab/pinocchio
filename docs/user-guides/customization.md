# Module Customization Guide

This comprehensive guide explains how to customize and extend various components of the Pinocchio system, including agents, plugins, tools, and workflows.

## Table of Contents

- [Agent Customization](#agent-customization)
- [Plugin Development](#plugin-development)
- [Tool Integration](#tool-integration)
- [Workflow Customization](#workflow-customization)
- [Memory & Knowledge Extensions](#memory--knowledge-extensions)
- [LLM Integration](#llm-integration)
- [Configuration Extensions](#configuration-extensions)
- [UI Customization](#ui-customization)

## Agent Customization

### Creating Custom Agents

All agents inherit from the `AgentWithRetry` base class. Here's how to create a custom agent:

#### 1. Basic Agent Structure

```python
from pinocchio.agents.base import AgentWithRetry
from pinocchio.data_models.agent import AgentResponse
from typing import Dict, Any

class CustomAgent(AgentWithRetry):
    """Custom agent for specialized tasks."""

    def __init__(self, llm_client: Any = None, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize custom agent."""
        if llm_client is None:
            config_manager = ConfigManager()
            agent_llm_config = config_manager.get_agent_llm_config("custom")
            llm_client = CustomLLMClient(agent_llm_config)

        super().__init__("custom", llm_client, max_retries, retry_delay)

        # Custom initialization
        self._setup_custom_capabilities()

    def _get_default_cuda_context(self) -> str:
        """Define specialized CUDA context for this agent."""
        return """You are a specialized CUDA expert focused on [specific domain].
        Your expertise includes:
        - [Capability 1]
        - [Capability 2]
        - [Capability 3]

        When processing requests:
        1. [Step 1]
        2. [Step 2]
        3. [Step 3]
        """

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """Execute custom agent logic."""
        request_id = request.get("request_id", "unknown")

        # Custom preprocessing
        processed_request = self._preprocess_request(request)

        # Build specialized prompt
        prompt = self._build_custom_prompt(processed_request)

        # Call LLM with retry
        llm_response = await self._call_llm_with_retry(prompt)

        # Process response
        output = self._process_custom_response(llm_response, processed_request)

        return self._create_response(
            request_id=request_id,
            success=True,
            output=output,
            processing_time_ms=int(self.get_average_processing_time())
        )

    def _setup_custom_capabilities(self):
        """Initialize custom capabilities."""
        # Custom tool integration
        # Custom memory/knowledge setup
        # Custom configuration loading
        pass

    def _preprocess_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Custom request preprocessing."""
        # Add custom logic here
        return request

    def _build_custom_prompt(self, request: Dict[str, Any]) -> str:
        """Build specialized prompt for this agent."""
        # Custom prompt engineering
        pass

    def _process_custom_response(self, llm_response: Dict[str, Any],
                               request: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM response with custom logic."""
        # Custom response processing
        pass
```

#### 2. Agent Registration

Add your custom agent to the coordinator:

```python
# In pinocchio/coordinator.py
from .agents.custom_agent import CustomAgent

class Coordinator:
    def _initialize_agents(self):
        """Initialize all agents including custom ones."""
        # ... existing agents ...

        # Add custom agent
        if self.config.get("agents.custom.enabled", False):
            self.agents["custom"] = CustomAgent()
```

#### 3. Configuration Support

Add configuration support in `pinocchio.json`:

```json
{
  "agents": {
    "custom": {
      "enabled": true,
      "max_retries": 3,
      "timeout": null,
      "custom_settings": {
        "specialized_feature": true,
        "custom_parameter": "value"
      }
    }
  }
}
```

### Customizing Existing Agents

#### 1. Extending Agent Capabilities

```python
class EnhancedDebuggerAgent(DebuggerAgent):
    """Enhanced debugger with additional capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add custom tools
        self._setup_additional_tools()

        # Add custom knowledge
        self._load_specialized_knowledge()

    def _setup_additional_tools(self):
        """Add specialized debugging tools."""
        from .tools.custom_debug_tools import CustomDebugTools
        CustomDebugTools.register_tools(self.tool_manager)

    def _run_debugging_tools(self, cuda_code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced debugging with additional tools."""
        # Run base debugging tools
        tool_results = super()._run_debugging_tools(cuda_code, context)

        # Add custom debugging logic
        custom_results = self._run_custom_analysis(cuda_code, context)
        tool_results.update(custom_results)

        return tool_results
```

#### 2. Custom Prompt Engineering

```python
class CustomGeneratorAgent(GeneratorAgent):
    """Generator with custom prompt templates."""

    def _build_generation_prompt(self, request: Dict[str, Any]) -> str:
        """Build custom generation prompt."""
        base_prompt = super()._build_generation_prompt(request)

        # Add domain-specific context
        domain_context = self._get_domain_context(request)

        # Apply custom templates
        if request.get("domain") == "machine_learning":
            template = self._get_ml_template()
            return f"{base_prompt}\n\n{template}\n\n{domain_context}"

        return base_prompt

    def _get_domain_context(self, request: Dict[str, Any]) -> str:
        """Get domain-specific context."""
        domain = request.get("domain", "general")

        domain_contexts = {
            "machine_learning": """
            Focus on ML-optimized CUDA kernels:
            - Tensor operations optimization
            - Memory layout for neural networks
            - Batch processing considerations
            """,
            "scientific_computing": """
            Focus on scientific computing patterns:
            - Numerical precision considerations
            - Large-scale data processing
            - Mathematical library integration
            """
        }

        return domain_contexts.get(domain, "")
```

## Plugin Development

### Creating Custom Plugins

#### 1. Plugin Base Classes

```python
from pinocchio.plugins.base import Plugin
from typing import Dict, Any

class CustomWorkflowPlugin(Plugin):
    """Custom workflow plugin."""

    def __init__(self):
        super().__init__(
            name="custom_workflow",
            description="Custom workflow for specialized tasks"
        )
        self.workflows = {}

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin with configuration."""
        try:
            workflow_configs = config.get("workflows", {})
            for name, workflow_config in workflow_configs.items():
                self.workflows[name] = self._parse_workflow(workflow_config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize custom workflow plugin: {e}")
            return False

    def execute(self, **kwargs) -> Any:
        """Execute custom workflow."""
        workflow_name = kwargs.get("workflow_name")
        request = kwargs.get("request", {})

        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        workflow = self.workflows[workflow_name]
        return self._execute_workflow(workflow, request)

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.workflows.clear()

    def _parse_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse workflow configuration."""
        # Custom workflow parsing logic
        return {
            "name": config["name"],
            "steps": config["steps"],
            "dependencies": config.get("dependencies", {}),
            "settings": config.get("settings", {})
        }

    def _execute_workflow(self, workflow: Dict[str, Any],
                         request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute custom workflow logic."""
        tasks = []

        for step in workflow["steps"]:
            task = {
                "id": f"custom_{step['id']}",
                "agent_type": step["agent"],
                "description": step["description"],
                "priority": step.get("priority", "medium"),
                "custom_params": step.get("params", {})
            }
            tasks.append(task)

        return tasks
```

#### 2. Agent Enhancement Plugin

```python
class AgentEnhancementPlugin(Plugin):
    """Plugin to enhance agent capabilities."""

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize agent enhancements."""
        self.enhancements = config.get("enhancements", {})
        return True

    def execute(self, agent_type: str, **kwargs) -> Dict[str, Any]:
        """Apply enhancements to agent."""
        if agent_type not in self.enhancements:
            return {}

        enhancement = self.enhancements[agent_type]

        return {
            "additional_context": enhancement.get("context", ""),
            "custom_instructions": enhancement.get("instructions", []),
            "specialized_knowledge": enhancement.get("knowledge", {}),
            "tool_configurations": enhancement.get("tools", {})
        }
```

#### 3. Plugin Registration

```python
# In pinocchio/plugins/__init__.py
from .custom_workflow_plugin import CustomWorkflowPlugin
from .agent_enhancement_plugin import AgentEnhancementPlugin

AVAILABLE_PLUGINS = {
    "custom_workflow": CustomWorkflowPlugin,
    "agent_enhancement": AgentEnhancementPlugin,
    # ... other plugins
}
```

### Plugin Configuration

Add plugin configuration to `pinocchio.json`:

```json
{
  "plugins": {
    "enabled": true,
    "plugins_directory": "./plugins",
    "active_plugins": {
      "workflow": "custom_workflow",
      "agent_enhancement": "agent_enhancement"
    },
    "plugin_configs": {
      "custom_workflow": {
        "workflows": {
          "custom_ml_workflow": {
            "name": "Machine Learning CUDA Workflow",
            "steps": [
              {
                "id": "generate_ml_kernel",
                "agent": "generator",
                "description": "Generate ML-optimized CUDA kernel",
                "params": {
                  "domain": "machine_learning",
                  "optimization_target": "throughput"
                }
              },
              {
                "id": "optimize_for_tensor_cores",
                "agent": "optimizer",
                "description": "Optimize for Tensor Core usage",
                "params": {
                  "target_hardware": "V100",
                  "precision": "mixed"
                }
              }
            ]
          }
        }
      },
      "agent_enhancement": {
        "enhancements": {
          "generator": {
            "context": "Focus on ML-specific optimizations",
            "instructions": [
              "Always consider tensor core utilization",
              "Optimize for batch processing"
            ]
          }
        }
      }
    }
  }
}
```

## Tool Integration

### Creating Custom MCP Tools

#### 1. Custom Debug Tool

```python
from pinocchio.tools.base import MCPTool, ToolResult, ToolExecutionStatus
from typing import Dict, Any

class CustomStaticAnalyzer(MCPTool):
    """Custom static analysis tool for CUDA code."""

    def __init__(self, timeout: int = 60):
        super().__init__(
            name="custom_static_analyzer",
            description="Custom static analysis for CUDA patterns",
            timeout=timeout
        )
        self._setup_analysis_rules()

    def execute(self, cuda_code: str, analysis_level: str = "standard") -> ToolResult:
        """Execute custom static analysis."""
        try:
            # Run custom analysis
            analysis_results = self._analyze_code(cuda_code, analysis_level)

            # Format results
            output = self._format_analysis_results(analysis_results)

            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                output=output,
                metadata={
                    'analysis_level': analysis_level,
                    'rules_applied': len(self.analysis_rules),
                    'issues_found': len(analysis_results['issues']),
                    'suggestions': analysis_results.get('suggestions', [])
                }
            )

        except Exception as e:
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                output="",
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA source code to analyze"
                },
                "analysis_level": {
                    "type": "string",
                    "enum": ["basic", "standard", "comprehensive"],
                    "description": "Level of analysis to perform",
                    "default": "standard"
                }
            },
            "required": ["cuda_code"]
        }

    def _setup_analysis_rules(self):
        """Setup custom analysis rules."""
        self.analysis_rules = [
            self._check_memory_coalescing,
            self._check_occupancy_patterns,
            self._check_register_usage,
            self._check_shared_memory_usage,
            self._check_custom_patterns
        ]

    def _analyze_code(self, code: str, level: str) -> Dict[str, Any]:
        """Perform custom code analysis."""
        results = {
            'issues': [],
            'suggestions': [],
            'metrics': {}
        }

        # Apply analysis rules based on level
        rules_to_apply = self.analysis_rules
        if level == "basic":
            rules_to_apply = self.analysis_rules[:2]
        elif level == "comprehensive":
            rules_to_apply = self.analysis_rules + self._get_advanced_rules()

        for rule in rules_to_apply:
            rule_result = rule(code)
            if rule_result:
                results['issues'].extend(rule_result.get('issues', []))
                results['suggestions'].extend(rule_result.get('suggestions', []))

        return results

    def _check_custom_patterns(self, code: str) -> Dict[str, Any]:
        """Check for custom patterns specific to your domain."""
        # Implement custom pattern checking
        pass
```

#### 2. Custom Performance Tool

```python
class CustomPerformanceProfiler(MCPTool):
    """Custom performance profiling tool."""

    def __init__(self, timeout: int = 120):
        super().__init__(
            name="custom_profiler",
            description="Custom performance profiling with ML optimizations",
            timeout=timeout
        )

    def execute(self, cuda_code: str, profile_type: str = "performance",
                target_hardware: str = "V100") -> ToolResult:
        """Execute custom profiling."""
        try:
            # Custom profiling logic
            profile_results = self._run_custom_profiling(
                cuda_code, profile_type, target_hardware
            )

            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                profile_results, target_hardware
            )

            output = self._format_profiling_results(profile_results, recommendations)

            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                output=output,
                metadata={
                    'profile_type': profile_type,
                    'target_hardware': target_hardware,
                    'performance_score': profile_results.get('score', 0),
                    'recommendations': recommendations
                }
            )

        except Exception as e:
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                output="",
                error=str(e)
            )

    def _run_custom_profiling(self, code: str, profile_type: str,
                            hardware: str) -> Dict[str, Any]:
        """Run custom profiling analysis."""
        # Implement custom profiling logic
        # This could integrate with external tools or use static analysis
        pass
```

#### 3. Tool Registration

```python
# Create custom tools container
class CustomTools:
    """Container for custom tools."""

    @staticmethod
    def get_all_tools() -> List[MCPTool]:
        """Get all custom tools."""
        return [
            CustomStaticAnalyzer(),
            CustomPerformanceProfiler(),
            # Add more custom tools
        ]

    @staticmethod
    def register_tools(tool_manager) -> None:
        """Register all custom tools."""
        tools = CustomTools.get_all_tools()
        for tool in tools:
            tool_manager.register_tool(tool)

# In agent initialization
class CustomDebuggerAgent(DebuggerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register custom tools
        CustomTools.register_tools(self.tool_manager)
```

## Workflow Customization

### Custom Workflow Definitions

#### 1. JSON-based Workflow

```json
{
  "workflows": {
    "custom_ml_pipeline": {
      "name": "Machine Learning CUDA Pipeline",
      "description": "Specialized workflow for ML kernel development",
      "tasks": [
        {
          "id": "analyze_requirements",
          "agent_type": "analyzer",
          "description": "Analyze ML-specific requirements",
          "requirements": {
            "input_analysis": true,
            "tensor_shape_analysis": true,
            "performance_requirements": true
          },
          "priority": "critical"
        },
        {
          "id": "generate_ml_kernel",
          "agent_type": "generator",
          "description": "Generate ML-optimized CUDA kernel",
          "requirements": {
            "language": "cuda",
            "optimization_target": "ml_workloads",
            "tensor_core_usage": true
          },
          "optimization_goals": ["throughput", "memory_efficiency"],
          "priority": "critical",
          "dependencies": ["analyze_requirements"]
        },
        {
          "id": "validate_tensor_operations",
          "agent_type": "debugger",
          "description": "Validate tensor operations and memory access",
          "requirements": {
            "tensor_validation": true,
            "memory_layout_check": true,
            "numerical_stability": true
          },
          "priority": "critical",
          "dependencies": ["generate_ml_kernel"]
        },
        {
          "id": "optimize_for_hardware",
          "agent_type": "optimizer",
          "description": "Optimize for specific ML hardware",
          "requirements": {
            "target_architecture": "compute_75",
            "tensor_core_optimization": true,
            "mixed_precision": true
          },
          "optimization_goals": ["tensor_core_utilization", "memory_bandwidth"],
          "priority": "high",
          "dependencies": ["validate_tensor_operations"]
        },
        {
          "id": "benchmark_performance",
          "agent_type": "evaluator",
          "description": "Benchmark ML performance metrics",
          "requirements": {
            "performance_metrics": ["flops", "memory_bandwidth", "tensor_core_utilization"],
            "ml_specific_metrics": true,
            "comparison_baselines": ["cuDNN", "PyTorch"]
          },
          "priority": "high",
          "dependencies": ["optimize_for_hardware"]
        }
      ]
    }
  }
}
```

#### 2. Python-based Workflow

```python
class CustomWorkflowGenerator:
    """Generate custom workflows dynamically."""

    def generate_workflow(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate workflow based on request analysis."""
        domain = request.get("domain", "general")
        complexity = request.get("complexity", "medium")

        if domain == "machine_learning":
            return self._generate_ml_workflow(request, complexity)
        elif domain == "scientific_computing":
            return self._generate_scientific_workflow(request, complexity)
        else:
            return self._generate_general_workflow(request, complexity)

    def _generate_ml_workflow(self, request: Dict[str, Any],
                            complexity: str) -> List[Dict[str, Any]]:
        """Generate ML-specific workflow."""
        tasks = []

        # Analysis phase
        tasks.append({
            "id": "ml_requirements_analysis",
            "agent_type": "analyzer",
            "description": "Analyze ML requirements and constraints",
            "custom_params": {
                "focus_areas": ["tensor_operations", "memory_patterns", "compute_intensity"],
                "optimization_targets": request.get("optimization_targets", ["performance"])
            }
        })

        # Generation phase with ML focus
        tasks.append({
            "id": "ml_kernel_generation",
            "agent_type": "generator",
            "description": "Generate ML-optimized CUDA kernel",
            "dependencies": ["ml_requirements_analysis"],
            "custom_params": {
                "template_category": "machine_learning",
                "tensor_core_preference": True,
                "mixed_precision": request.get("mixed_precision", False)
            }
        })

        # ML-specific validation
        if complexity in ["medium", "high"]:
            tasks.append({
                "id": "ml_validation",
                "agent_type": "debugger",
                "description": "Validate ML-specific constraints",
                "dependencies": ["ml_kernel_generation"],
                "custom_params": {
                    "validation_categories": ["numerical_stability", "tensor_shapes", "memory_layout"]
                }
            })

        return tasks
```

### Workflow Plugin Integration

```python
class DynamicWorkflowPlugin(Plugin):
    """Plugin for dynamic workflow generation."""

    def __init__(self):
        super().__init__(
            name="dynamic_workflow",
            description="Generate workflows dynamically based on request analysis"
        )
        self.workflow_generator = CustomWorkflowGenerator()

    def execute(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate custom workflow."""
        # Analyze request to determine workflow type
        analysis = self._analyze_request(request)

        # Generate appropriate workflow
        workflow = self.workflow_generator.generate_workflow(analysis)

        return workflow

    def _analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request to determine workflow requirements."""
        # Implement request analysis logic
        pass
```

## Memory & Knowledge Extensions

### Custom Memory Providers

```python
from pinocchio.memory.manager import MemoryManager

class CustomMemoryProvider:
    """Custom memory provider with specialized indexing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_custom_storage()

    def store_memory(self, agent_id: str, session_id: str,
                    memory_data: Dict[str, Any]) -> str:
        """Store memory with custom indexing."""
        # Custom storage logic
        memory_id = self._generate_memory_id(agent_id, session_id)

        # Add custom metadata
        enhanced_data = self._enhance_memory_data(memory_data)

        # Store with custom indexing
        self._store_with_custom_index(memory_id, enhanced_data)

        return memory_id

    def query_memories(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query memories with custom search logic."""
        # Implement custom search algorithm
        pass

    def _enhance_memory_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add custom metadata to memory data."""
        enhanced = data.copy()
        enhanced.update({
            'custom_tags': self._extract_custom_tags(data),
            'semantic_features': self._extract_semantic_features(data),
            'relevance_scores': self._calculate_relevance_scores(data)
        })
        return enhanced
```

### Custom Knowledge Sources

```python
class CustomKnowledgeProvider:
    """Custom knowledge provider for specialized domains."""

    def __init__(self, domain: str):
        self.domain = domain
        self.knowledge_base = self._load_domain_knowledge()

    def query_knowledge(self, keywords: List[str],
                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query domain-specific knowledge."""
        # Custom knowledge retrieval logic
        relevant_fragments = []

        for keyword in keywords:
            fragments = self._search_domain_knowledge(keyword, context)
            relevant_fragments.extend(fragments)

        # Rank and filter results
        ranked_fragments = self._rank_by_relevance(relevant_fragments, context)

        return ranked_fragments[:10]  # Return top 10

    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain-specific knowledge base."""
        # Load knowledge from various sources
        knowledge = {}

        if self.domain == "machine_learning":
            knowledge.update(self._load_ml_knowledge())
        elif self.domain == "scientific_computing":
            knowledge.update(self._load_scientific_knowledge())

        return knowledge

    def _load_ml_knowledge(self) -> Dict[str, Any]:
        """Load ML-specific CUDA knowledge."""
        return {
            "tensor_operations": {
                "description": "Optimized tensor operation patterns",
                "patterns": [
                    "Matrix multiplication with tensor cores",
                    "Convolution optimization techniques",
                    "Batch normalization kernels"
                ],
                "best_practices": [
                    "Use mixed precision for tensor cores",
                    "Optimize memory layout for tensor operations",
                    "Consider warp-level primitives"
                ]
            },
            # Add more ML-specific knowledge
        }
```

## LLM Integration

### Custom LLM Providers

```python
from pinocchio.llm.base import BaseLLMClient

class CustomLLMProvider(BaseLLMClient):
    """Custom LLM provider integration."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_endpoint = config["custom_endpoint"]
        self.custom_headers = config.get("custom_headers", {})

    async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Custom completion implementation."""
        try:
            # Prepare custom request
            request_data = self._prepare_custom_request(prompt, **kwargs)

            # Make API call to custom provider
            response = await self._make_custom_api_call(request_data)

            # Process custom response format
            parsed_response = self._parse_custom_response(response)

            return parsed_response

        except Exception as e:
            logger.error(f"Custom LLM provider error: {e}")
            raise

    def _prepare_custom_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare request for custom LLM provider."""
        # Custom request formatting
        return {
            "input": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "custom_param": kwargs.get("custom_param", "default")
            }
        }

    async def _make_custom_api_call(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to custom provider."""
        # Implement custom API integration
        pass

    def _parse_custom_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse custom provider response format."""
        # Convert custom format to standard format
        return {
            "agent_type": response.get("agent_type", "unknown"),
            "success": response.get("success", True),
            "output": response.get("output", {}),
            "explanation": response.get("explanation", ""),
            "confidence": response.get("confidence", 0.95)
        }
```

### LLM Provider Registration

```python
# In pinocchio/llm/__init__.py
from .custom_provider import CustomLLMProvider

LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "custom": CustomLLMClient,
    "custom_provider": CustomLLMProvider,  # Add custom provider
}

def get_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    """Get LLM client based on configuration."""
    provider = config.get("provider", "openai")

    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return LLM_PROVIDERS[provider](config)
```

## Configuration Extensions

### Custom Configuration Sections

```python
# Custom configuration model
from pydantic import BaseModel
from typing import Optional, Dict, Any

class CustomConfig(BaseModel):
    """Custom configuration section."""
    enabled: bool = True
    custom_settings: Dict[str, Any] = {}
    advanced_features: Optional[Dict[str, Any]] = None

class ExtendedPinocchioConfig(PinocchioConfig):
    """Extended configuration with custom sections."""
    custom: Optional[CustomConfig] = None
    domain_specific: Optional[Dict[str, Any]] = None

# Usage in configuration loading
def load_extended_config(config_path: str) -> ExtendedPinocchioConfig:
    """Load extended configuration."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    return ExtendedPinocchioConfig(**config_data)
```

### Environment-based Configuration

```python
class EnvironmentConfigManager(ConfigManager):
    """Configuration manager with environment-specific overrides."""

    def __init__(self, config_path: str = "pinocchio.json"):
        super().__init__(config_path)
        self._apply_environment_overrides()

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        env = os.getenv("PINOCCHIO_ENV", "development")

        env_configs = {
            "development": self._get_development_config(),
            "testing": self._get_testing_config(),
            "production": self._get_production_config()
        }

        if env in env_configs:
            self._merge_config(env_configs[env])

    def _get_development_config(self) -> Dict[str, Any]:
        """Get development-specific configuration."""
        return {
            "verbose": {"enabled": True, "level": "maximum"},
            "debug_repair": {"max_repair_attempts": 5},
            "tools": {"enabled": True}
        }

    def _get_production_config(self) -> Dict[str, Any]:
        """Get production-specific configuration."""
        return {
            "verbose": {"enabled": False},
            "logging": {"level": "WARNING"},
            "tools": {"enabled": False}  # Disable tools in production
        }
```

## UI Customization

### Custom CLI Interface

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

class CustomCLI:
    """Custom CLI interface with specialized displays."""

    def __init__(self):
        self.console = Console()
        self.setup_custom_themes()

    def setup_custom_themes(self):
        """Setup custom visual themes."""
        self.themes = {
            "ml_workflow": {
                "primary_color": "deep_sky_blue1",
                "secondary_color": "spring_green1",
                "accent_color": "gold1"
            },
            "debug_mode": {
                "primary_color": "red1",
                "secondary_color": "orange1",
                "accent_color": "yellow1"
            }
        }

    def display_custom_workflow_progress(self, workflow_name: str, tasks: List[Dict[str, Any]]):
        """Display custom workflow progress."""
        theme = self.themes.get("ml_workflow", {})

        # Create custom table
        table = Table(title=f"🤖 {workflow_name} Progress",
                     title_style=theme.get("primary_color", "blue"))

        table.add_column("Step", style=theme.get("secondary_color", "cyan"))
        table.add_column("Agent", style=theme.get("accent_color", "yellow"))
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for i, task in enumerate(tasks, 1):
            status_icon = "🟢" if task["status"] == "completed" else "🟡"
            table.add_row(
                str(i),
                task["agent_type"].upper(),
                f"{status_icon} {task['status']}",
                task.get("description", "")
            )

        self.console.print(table)

    def display_domain_specific_results(self, domain: str, results: Dict[str, Any]):
        """Display domain-specific results."""
        if domain == "machine_learning":
            self._display_ml_results(results)
        elif domain == "scientific_computing":
            self._display_scientific_results(results)
        else:
            self._display_general_results(results)

    def _display_ml_results(self, results: Dict[str, Any]):
        """Display ML-specific results with custom formatting."""
        panel_content = []

        if "tensor_operations" in results:
            panel_content.append("🧮 **Tensor Operations Optimized**")
            for op in results["tensor_operations"]:
                panel_content.append(f"  • {op}")

        if "performance_metrics" in results:
            metrics = results["performance_metrics"]
            panel_content.append("\n📊 **Performance Metrics**")
            panel_content.append(f"  • Tensor Core Utilization: {metrics.get('tensor_core_util', 'N/A')}")
            panel_content.append(f"  • Memory Efficiency: {metrics.get('memory_efficiency', 'N/A')}")

        self.console.print(Panel(
            "\n".join(panel_content),
            title="🤖 ML Optimization Results",
            border_style="deep_sky_blue1"
        ))
```

### Custom Web Interface

```python
from flask import Flask, request, jsonify, render_template

class CustomWebInterface:
    """Custom web interface for specialized workflows."""

    def __init__(self, coordinator):
        self.app = Flask(__name__)
        self.coordinator = coordinator
        self.setup_custom_routes()

    def setup_custom_routes(self):
        """Setup custom web routes."""

        @self.app.route('/ml-workflow', methods=['GET', 'POST'])
        def ml_workflow():
            """ML-specific workflow interface."""
            if request.method == 'POST':
                ml_request = self._parse_ml_request(request.json)
                result = self.coordinator.process_request(ml_request)
                return jsonify(result)

            return render_template('ml_workflow.html')

        @self.app.route('/scientific-computing', methods=['GET', 'POST'])
        def scientific_workflow():
            """Scientific computing workflow interface."""
            if request.method == 'POST':
                sci_request = self._parse_scientific_request(request.json)
                result = self.coordinator.process_request(sci_request)
                return jsonify(result)

            return render_template('scientific_workflow.html')

        @self.app.route('/custom-analysis/<domain>')
        def custom_analysis(domain):
            """Custom analysis interface for specific domains."""
            return render_template('custom_analysis.html', domain=domain)

    def _parse_ml_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ML-specific request data."""
        return {
            "domain": "machine_learning",
            "task": data.get("task", ""),
            "tensor_shapes": data.get("tensor_shapes", {}),
            "optimization_targets": data.get("optimization_targets", ["performance"]),
            "hardware_constraints": data.get("hardware_constraints", {}),
            "workflow": "custom_ml_pipeline"
        }
```

## Best Practices for Customization

### 1. Modular Design
- Keep customizations in separate modules
- Use dependency injection for configuration
- Implement proper interfaces and inheritance

### 2. Configuration Management
- Use environment-specific configurations
- Validate custom configuration sections
- Provide sensible defaults

### 3. Error Handling
- Implement proper error handling in custom components
- Provide fallback mechanisms
- Log custom component activities

### 4. Testing
- Write tests for custom components
- Test integration with existing system
- Validate custom configurations

### 5. Documentation
- Document custom components and their usage
- Provide configuration examples
- Include migration guides for updates

---

This customization guide provides a comprehensive foundation for extending and customizing the Pinocchio system to meet specific requirements and domains.
