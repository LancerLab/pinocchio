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

## 🚀 核心特性

- **简洁架构**：清晰的模块职责和通信路径
- **流式体验**：实时进度反馈，类似聊天AI的用户体验
- **完整记录**：结构化日志记录，便于调试和分析
- **模块化设计**：松耦合架构，便于扩展和维护
- **易于调试**：JSON文件存储，便于查看和调试

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Coordinator    │───▶│  Planning Agent │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SessionLogger  │◀───│  PromptManager  │◀───│  Plan: TodoList │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MemoryManager  │◀───│     Agent       │───▶│      LLM        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ KnowledgeManager│
                       └─────────────────┘
```

### 核心组件

| 组件 | 职责 | 核心功能 |
|------|------|----------|
| **Coordinator** | 系统总指挥 | 流程控制、Session管理、流式输出 |
| **SessionLogger** | 结构化logger | 摘要日志、详细通信记录、持久化 |
| **PromptManager** | 综合prompt构建器 | 整合Memory+Knowledge+Context |
| **Agent** | 纯执行器 | 调用LLM、解析结构化输出 |
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
├── session_logger.py      # 结构化logger - 会话管理
├── agents/               # 智能体模块
│   ├── __init__.py
│   ├── base.py           # 智能体基类
│   ├── generator.py      # 代码生成智能体
│   └── planner.py        # 规划智能体
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
│   └── agent.py         # 智能体数据模型
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
    }
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
