# Pinocchio - 多智能体协作系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pinocchio 是一个用于自动编写、调试、优化 Choreo 计算核编程 DSL 算子的多智能体协作系统。系统采用简洁的架构设计，以 Coordinator 为总指挥，Session 为结构化 logger，支持流式输出和完整的交互历史记录。

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

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-org/pinocchio.git
cd pinocchio

# 使用 Poetry 安装
poetry install

# 或使用 pip 安装
pip install -e .
```

## 🚀 快速开始

### 基础使用

```python
from pinocchio.coordinator import Coordinator

# 创建协调器
coordinator = Coordinator()

# 处理用户请求
async def main():
    async for message in coordinator.process_user_request("编写一个conv 2d算子"):
        print(message)  # 流式输出进度信息

# 运行
import asyncio
asyncio.run(main())
```

### 命令行使用

```bash
# 基础使用
python -m pinocchio --prompt "编写一个conv 2d算子"

# 指定配置文件
python -m pinocchio --config config.json --prompt "优化现有的算子"
```

## 📁 项目结构

```
pinocchio/
├── coordinator.py          # 总指挥
├── session_logger.py      # 结构化logger
├── prompt_manager.py      # 综合prompt构建器
├── agent.py              # Agent基类和实现
├── memory_manager.py     # 记忆管理
├── knowledge_manager.py  # 知识管理
├── llm_client.py        # LLM客户端
├── models/              # 数据模型
│   ├── __init__.py
│   ├── session.py
│   ├── memory.py
│   ├── knowledge.py
│   └── agent.py
└── utils/               # 工具函数
    ├── __init__.py
    ├── json_parser.py
    └── file_utils.py

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

### 添加新的Agent

```python
from pinocchio.agent import Agent

class CustomAgent(Agent):
    def __init__(self, agent_type: str, llm_client):
        super().__init__(agent_type, llm_client)

    async def execute(self, prompt: Dict) -> Dict:
        # 实现自定义逻辑
        result = await self.llm_client.complete(prompt["prompt_string"])
        return self._parse_llm_response(result)
```

### 扩展知识库

```python
from pinocchio.knowledge_manager import KnowledgeManager

# 添加知识条目
knowledge_manager = KnowledgeManager()
knowledge_manager.add_knowledge({
    "id": "custom_knowledge",
    "agent_type": "generator",
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
