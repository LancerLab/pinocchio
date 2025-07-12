# Pinocchio - 多智能体协作系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pinocchio 是一个用于自动编写、调试、优化 Choreo 计算核编程 DSL 算子的多智能体协作系统。系统采用简洁的架构设计，以 Coordinator 为总指挥，Session 为结构化 logger，支持流式输出和完整的交互历史记录。

## 🎭 交互式 CLI 体验

Pinocchio 提供了丰富的交互式命令行界面，支持实时流式输出和完整的会话管理。以下是 MVP 典型需求的完整交互流程：

### 启动界面
```
╭─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                     │
│  🎭 Pinocchio CLI                                                                                   │
│                                                                                                     │
│  Welcome to Pinocchio - Multi-Agent Collaboration System                                            │
│                                                                                                     │
│  Type your request and press Enter to start...                                                      │
│  Type 'help' for available commands                                                                 │
│  Type 'quit' to exit                                                                                │
│                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯
>
```

### MVP 典型需求交互流程

**用户输入**: `编写一个矩阵加法算子`

**系统响应**:
```
👤 You: 编写一个矩阵加法算子

🤖 Pinocchio: [session_f8ffb8ae] Session started
🤖 Pinocchio: [session_f8ffb8ae] Plan generated: 1 steps
🤖 Pinocchio: [session_f8ffb8ae] Executing step 1: generator
🤖 Pinocchio: [session_f8ffb8ae] Step 1 completed successfully

🤖 Pinocchio:
🎉 Code generation completed!

```choreo
// Generated Choreo DSL operator for convolution operation
func conv_kernel(input: tensor, output: tensor) {
    // Basic implementation
    for i in range(input.shape[0]) {
        for j in range(input.shape[1]) {
            output[i][j] = compute(input[i][j]);
        }
    }
}
```

🤖 Pinocchio:
📋 Explanation: Generated a basic Choreo DSL operator with optimized memory access patterns.

🤖 Pinocchio:
⚡ Optimizations applied: loop_tiling, memory_coalescing

🤖 Pinocchio: [session_f8ffb8ae] Session completed successfully
🤖 Pinocchio: [session_f8ffb8ae] Session saved to: session_f8ffb8ae_20250711_180521.json
```

### 核心特性展示

- **🎭 美观界面**: 使用 Rich 库构建的现代化 CLI 界面
- **📊 实时反馈**: 流式输出显示每个步骤的进度和状态
- **🤖 多智能体协作**: 自动规划、执行和优化代码生成流程
- **💾 完整记录**: 自动保存会话日志，支持历史查询和调试
- **⚡ 智能优化**: 自动应用性能优化技术（循环分块、内存合并等）

## 🧠 智能任务规划系统

Pinocchio 的核心创新在于其智能任务规划机制，能够自动分解复杂需求并动态调整执行策略。

### 多轮优化链

系统支持多轮生成→调试→优化循环，每轮都包含完整的代码生成、错误检测和性能优化流程：

```
Round 1: Generator → Debugger → Optimizer
Round 2: Generator → Debugger → Optimizer
Round 3: Generator → Debugger → Optimizer
...
```

### 动态调试插入

当检测到编译错误或运行时问题时，系统会自动插入调试任务：

```
原始计划: Generator → Optimizer
检测到错误 → 动态插入: Generator → Debugger → Optimizer
```

### 实时任务可视化

系统提供实时的任务计划可视化界面，显示每个任务的执行状态和依赖关系：

```text
                                       Todolist (Task Plan)
  #  Task Description                                         Agent      Status        Depends On
  1  [Round 1] write a matmul for me                          generator  🟢 completed  -
  2  [Round 1] Compile and debug generated code               debugger   🟢 completed  task_1
  3  [Round 1] Optimise code for: performance and efficiency  optimizer  🟢 completed  task_2
  4  [Round 2] write a matmul for me                          generator  🟢 completed  task_3
  5  [Round 2] Compile and debug generated code               debugger   🟢 completed  task_4
  6  [Round 2] Optimise code for: performance and efficiency  optimizer  🟢 completed  task_5
  7  [Round 3] write a matmul for me                          generator  🟢 completed  task_6
  8  [Round 3] Compile and debug generated code               debugger   🟢 completed  task_7
  9  [Round 3] Optimise code for: performance and efficiency  optimizer  🟡 pending    task_8
 10  [Round 2] Continue code generation after bug fix         generator  🟢 completed  task_2
 11  [Round 2] Compile and debug generated code               debugger   🟢 completed  task_10
 12  [Round 2] Optimise code after bug fix                    optimizer  🟢 completed  task_11
 13  [Round 3] Continue code generation after bug fix         generator  🟢 completed  task_11
 14  [Round 3] Compile and debug generated code               debugger   🟢 completed  task_13
 15  [Round 3] Optimise code after bug fix                    optimizer  🟢 completed  task_14
 16  [Round 3] Continue code generation after bug fix         generator  🟢 completed  task_5
 17  [Round 3] Compile and debug generated code               debugger   🟢 completed  task_16
 18  [Round 3] Optimise code after bug fix                    optimizer  🟡 pending    task_17
```

> **说明**：每一轮任务链条（生成→调试→优化）自动串联，遇到错误时动态插入调试与修复任务，所有任务依赖关系、状态（🟢已完成/🟡待处理）一目了然，便于追踪和分析。

### 智能配置管理

系统通过配置文件控制优化行为：

```json
{
  "debug_repair": {
    "max_repair_attempts": 3
  },
  "optimization": {
    "max_optimisation_rounds": 3,
    "optimizer_enabled": true
  }
}
```

### 详细执行反馈

每个任务执行时提供详细的指令和状态反馈：

```
🤖 Pinocchio: [session_16763a26] 🔄 Executing 🔧 DEBUGGER (Task task_2)
🤖 Pinocchio: [session_16763a26]    📋 Description: [Round 1] Compile and debug generated code
🤖 Pinocchio: [session_16763a26]    💡 Detailed Instruction:
🤖 Pinocchio: [session_16763a26]       Compile and analyze the generated Choreo DSL code for errors.
🤖 Pinocchio: [session_16763a26]       Debugging Goals:
🤖 Pinocchio: [session_16763a26]       - Identify compilation errors
🤖 Pinocchio: [session_16763a26]       - Detect runtime issues
🤖 Pinocchio: [session_16763a26]       - Provide detailed error analysis
🤖 Pinocchio: [session_16763a26]       - Suggest fixes and improvements
```

### 智能体参与统计

系统提供详细的智能体参与统计信息：

```
🤖 Pinocchio: [session_16763a26] 🤖 Agent Participation Summary:
🤖 Pinocchio: [session_16763a26]    ⚡ GENERATOR: 6/6 (100.0% success)
🤖 Pinocchio: [session_16763a26]    🔧 DEBUGGER: 6/6 (100.0% success)
🤖 Pinocchio: [session_16763a26]    🚀 OPTIMIZER: 6/6 (100.0% success)
```

## 🚀 核心特性

- **简洁架构**：清晰的模块职责和通信路径
- **流式体验**：实时进度反馈，类似聊天AI的用户体验
- **完整记录**：结构化日志记录，便于调试和分析
- **模块化设计**：松耦合架构，便于扩展和维护
- **易于调试**：JSON文件存储，便于查看和调试
- **智能任务规划**：自动分解复杂需求，动态调整执行策略
- **多轮优化**：支持多轮生成→调试→优化循环
- **动态调试插入**：根据错误自动插入调试任务
- **实时可视化**：任务计划的可视化界面和状态跟踪

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Coordinator    │───▶│  TaskPlanner    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SessionLogger  │◀───│  TaskExecutor   │◀───│  Task Plan      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MemoryManager  │◀───│  PromptManager  │◀───│  Agent Pool     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ KnowledgeManager│    │      LLM        │
                       └─────────────────┘    └─────────────────┘
```

### 核心组件

| 组件 | 职责 | 核心功能 |
|------|------|----------|
| **Coordinator** | 系统总指挥 | 流程控制、Session管理、流式输出 |
| **TaskPlanner** | 任务规划器 | 智能任务分解、多轮优化链生成 |
| **TaskExecutor** | 任务执行器 | 动态任务调度、错误恢复、依赖管理 |
| **SessionLogger** | 结构化logger | 摘要日志、详细通信记录、持久化 |
| **PromptManager** | 综合prompt构建器 | 整合Memory+Knowledge+Context |
| **Agent Pool** | 智能体池 | Generator、Debugger、Optimizer管理 |
| **MemoryManager** | 记忆管理 | 存储Agent交互、检索相关记忆 |
| **KnowledgeManager** | 知识管理 | 只读知识、按需检索 |
| **LLM** | 大语言模型接口 | 统一的LLM调用封装 |

## 📦 安装

### 环境要求

- Python 3.9+
- Poetry (推荐) 或 pip
- uv (可选，极快的包管理器，需单独安装)

### 安装 uv（可选）

```bash
# 推荐使用 pip 安装 uv
pip install uv
# 或使用官方安装脚本
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-org/pinocchio.git
cd pinocchio

# 使用 uv 安装依赖（推荐，极快）
uv pip install -r requirements.txt

# 或使用 Poetry 安装
poetry install

# 设置开发环境（包含pre-commit钩子）
./scripts/setup_dev.sh

# 或使用 pip 安装
pip install -e .
```

### 开发环境设置

为了确保pre-commit钩子正常工作，请运行开发环境设置脚本：

```bash
# 自动设置开发环境
./scripts/setup_dev.sh
```

这个脚本会：
- 安装所有Poetry依赖
- 安装pre-commit钩子
- 安装pre-commit需要的额外依赖
- 清理并重新安装钩子

## 🚀 快速开始

### 交互式 CLI 使用（推荐）

```bash
# 启动交互式 CLI
python -m pinocchio.cli.main

# 在 CLI 中输入你的需求
> 编写一个矩阵加法算子
> 优化现有的卷积算子
> 调试内存访问问题
```

### 程序化使用

```python
from pinocchio.coordinator import Coordinator

# 创建协调器
coordinator = Coordinator()

# 处理用户请求
async def main():
    async for message in coordinator.process_user_request("编写一个矩阵加法算子"):
        print(message)  # 流式输出进度信息

# 运行
import asyncio
asyncio.run(main())
```

### 直接命令行使用

```bash
# 单次请求处理
echo "编写一个矩阵加法算子" | python -m pinocchio.cli.main

# 或者使用 Python 模块
python -c "
import asyncio
from pinocchio.coordinator import Coordinator

async def main():
    coordinator = Coordinator()
    async for msg in coordinator.process_user_request('编写一个矩阵加法算子'):
        print(msg)

asyncio.run(main())
"
```

## 📁 项目结构

```
pinocchio/
├── coordinator.py          # 总指挥 - 多智能体协作核心
├── task_planner.py        # 智能任务规划器
├── task_executor.py       # 任务执行器
├── session_logger.py      # 结构化logger - 会话管理
├── agents/               # 智能体模块
│   ├── __init__.py
│   ├── base.py           # 智能体基类
│   ├── generator.py      # 代码生成智能体
│   ├── debugger.py       # 调试智能体
│   └── optimizer.py      # 优化智能体
├── cli/                  # 命令行界面
│   ├── __init__.py
│   └── main.py          # CLI 主程序
├── memory/               # 记忆管理
│   ├── __init__.py
│   ├── manager.py        # 记忆管理器
│   └── models/          # 记忆数据模型
├── session/              # 会话管理
│   ├── __init__.py
│   └── manager.py        # 会话管理器
├── llm/                  # LLM 客户端
│   ├── __init__.py
│   └── mock_client.py    # Mock LLM 客户端
├── prompt/               # 提示词管理
│   ├── __init__.py
│   └── models/          # 提示词模型
├── data_models/          # 数据模型
│   ├── __init__.py
│   ├── agent.py         # 智能体数据模型
│   └── task.py          # 任务数据模型
└── utils/               # 工具函数
    ├── __init__.py
    ├── file_utils.py    # 文件操作工具
    └── json_parser.py   # JSON 解析工具

# 数据存储目录
sessions/               # Session日志文件
memories/              # Memory存储
knowledge/             # Knowledge存储
```

## 🔧 配置

### 环境变量

```bash
# LLM API配置
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# 系统配置
PINOCCHIO_LOG_LEVEL=INFO
PINOCCHIO_STORAGE_PATH=./data
```

### 配置文件

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.7
  },
  "storage": {
    "sessions_path": "./sessions",
    "memories_path": "./memories",
    "knowledge_path": "./knowledge"
  },
  "agents": {
    "generator": {
      "enabled": true,
      "max_retries": 3
    },
    "debugger": {
      "enabled": true,
      "max_retries": 3
    },
    "optimizer": {
      "enabled": true,
      "max_retries": 3
    }
  },
  "debug_repair": {
    "max_repair_attempts": 3
  },
  "optimization": {
    "max_optimisation_rounds": 3,
    "optimizer_enabled": true
  }
}
```

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_coordinator.py

# 运行集成测试
pytest tests/integrations/

# 生成覆盖率报告
pytest --cov=pinocchio --cov-report=html
```

### 测试覆盖率

- 单元测试覆盖率 > 90%
- 集成测试覆盖主要工作流程
- 性能测试确保响应时间 < 30秒

## 📚 开发指南

### 添加新的智能体

```python
from pinocchio.agents.base import Agent

class CustomAgent(Agent):
    def __init__(self, agent_type: str, llm_client):
        super().__init__(agent_type, llm_client)

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        # 实现自定义逻辑
        prompt = self._build_prompt(request)
        result = await self._call_llm(prompt)
        return self._create_response(
            request_id=request["request_id"],
            success=True,
            output=result
        )

    def _get_agent_instructions(self) -> str:
        return "You are a custom agent specialized in..."

    def _get_output_format(self) -> str:
        return """
        Please provide your response in JSON format:
        {
            "agent_type": "custom",
            "success": true,
            "output": {
                // Custom output fields
            }
        }
        """
```

### 自定义任务规划策略

```python
from pinocchio.task_planner import TaskPlanner
from pinocchio.data_models.task import Task, TaskStatus, AgentType

class CustomTaskPlanner(TaskPlanner):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def generate_plan(self, user_request: str) -> List[Task]:
        """生成自定义任务计划"""
        tasks = []

        # 添加自定义任务
        tasks.append(Task(
            task_id=f"task_{len(tasks) + 1}",
            description="Custom analysis task",
            agent_type=AgentType.GENERATOR,
            priority=1,
            dependencies=[],
            status=TaskStatus.PENDING
        ))

        return tasks
```

### 扩展任务执行逻辑

```python
from pinocchio.task_executor import TaskExecutor

class CustomTaskExecutor(TaskExecutor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    async def _execute_task(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        """自定义任务执行逻辑"""
        # 实现自定义执行逻辑
        result = await super()._execute_task(task, context)

        # 添加自定义后处理
        if result.success and task.agent_type == AgentType.GENERATOR:
            # 自定义生成后处理
            pass

        return result
```

### 扩展记忆管理

```python
from pinocchio.memory.manager import MemoryManager

# 创建记忆管理器
memory_manager = MemoryManager()

# 添加记忆条目
memory_manager.add_memory({
    "agent_type": "generator",
    "task_description": "矩阵加法算子",
    "output": {"code": "...", "optimizations": [...]},
    "success": True
})

# 检索相关记忆
related_memories = memory_manager.search_memories("矩阵加法")
```
    "keywords": ["custom", "algorithm"],
    "content": "自定义算法知识...",
    "category": "algorithm"
})
```

## 🤝 贡献指南

### 开发环境设置

```bash
# 安装开发依赖
poetry install --with dev

# 安装预提交钩子
pre-commit install

# 运行代码检查
pre-commit run --all-files
```

### 提交规范

- 使用 [Conventional Commits](https://www.conventionalcommits.org/)
- 每个提交都应该有清晰的描述
- 包含相关的测试用例

### 代码规范

- 遵循 PEP 8 代码风格
- 使用 Black 进行代码格式化
- 使用 isort 排序导入
- 所有函数添加类型注解

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户。

## 📞 联系我们

- 问题反馈：[GitHub Issues](https://github.com/your-org/pinocchio/issues)
- 功能建议：[GitHub Discussions](https://github.com/your-org/pinocchio/discussions)
- 邮件联系：pinocchio@example.com

---

**注意**：本项目仍在积极开发中，API 可能会发生变化。请查看 [CHANGELOG.md](CHANGELOG.md) 了解最新更新。
