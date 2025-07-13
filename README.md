# Pinocchio - 多智能体协作系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-396%20passed%2C%209%20skipped-brightgreen.svg)](https://github.com/your-repo/pinocchio)
[![Test Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/your-repo/pinocchio)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-repo/pinocchio)

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
- **Utility模块**：统一的工具函数库，提供文件操作、JSON解析、配置管理等核心功能
- **LLM连接测试**：独立的LLM连接测试脚本，支持手动验证API密钥和连接状态

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
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Knowledge     │◀───│   Utils         │◀───│  LLM Client     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Utility模块

Pinocchio 提供了统一的工具函数库 (`@/utils`)，为各个核心模块提供标准化的工具函数：

### 核心功能
- **文件操作**: 安全的文件读写、目录管理、临时文件处理
- **JSON解析**: 结构化输出解析、代码提取、响应验证
- **配置管理**: 测试配置创建、配置合并、配置验证
- **临时文件**: 临时文件/目录创建、清理、路径管理

### 使用示例
```python
from pinocchio.utils import (
    safe_write_json, safe_read_json,
    parse_structured_output, extract_code_from_response,
    create_temp_file, cleanup_temp_files
)

# 安全的JSON操作
safe_write_json("data.json", {"key": "value"})
data = safe_read_json("data.json")

# 结构化输出解析
result = parse_structured_output(llm_response)

# 临时文件管理
temp_file = create_temp_file("test_", ".txt")
cleanup_temp_files()
```

详细设计文档请参考：[Utility模块设计文档](docs/development/utility_module_design.md)

## 📊 测试状态

- **总测试数**: 405 (396通过, 9跳过)
- **测试覆盖率**: 95%
- **快速测试模式**: 支持 `FAST_TEST=1` 环境变量
- **测试执行时间**: ~10秒 (优化后)

### 测试运行方式
```bash
# 完整测试
python -m pytest tests/ -v

# 快速测试 (跳过慢速测试)
./scripts/run_fast_tests.sh

# 跳过真实LLM连接测试
python -m pytest tests/ -m "not real_llm" -v
```

## 🔗 LLM连接测试

**注意**: 真实的LLM连接测试不包含在常规单元测试中，以避免依赖外部服务和产生费用。

### 手动测试LLM连接
```bash
# 快速连接测试
python scripts/test_llm_connection.py

# 完整健康检查
python scripts/health_check.py --all

# 测试特定提供商
python scripts/test_llm_connection.py --provider openai
```

### 环境变量设置
```bash
export OPENAI_API_KEY="your-api-key-here"
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-3.5-turbo"
```

详细说明请参考：[scripts/README.md](scripts/README.md)

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行测试
```bash
# 快速测试
./scripts/run_fast_tests.sh

# 完整测试
python -m pytest tests/ -v
```

### 启动CLI
```bash
python -m pinocchio.cli
```

## 📚 文档

- [API文档](docs/api/)
- [开发指南](docs/development/)
- [测试指南](docs/testing/)
- [Utility模块设计](docs/development/utility_module_design.md)
- [测试性能优化](docs/development/test_performance_optimization.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。
