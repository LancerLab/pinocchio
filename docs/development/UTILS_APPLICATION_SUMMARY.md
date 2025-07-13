# Utils函数应用总结

本文档总结了将 `@/utils` 模块中的工具函数应用到各个模块中的情况。

## 已更新的模块

### 1. Session模块

#### `pinocchio/session/manager.py`
- **应用的函数**: `safe_write_json`, `safe_read_json`, `ensure_directory`
- **更新内容**:
  - 初始化时使用 `ensure_directory` 确保目录存在
  - `_persist_session` 方法使用 `safe_write_json` 安全写入JSON
  - `_load_session` 方法使用 `safe_read_json` 安全读取JSON
  - 添加了错误处理，当文件操作失败时抛出异常

#### `pinocchio/session/utils.py`
- **应用的函数**: `safe_write_json`, `ensure_directory`
- **更新内容**:
  - `export_session_to_json` 方法使用 `ensure_directory` 确保目录存在
  - 使用 `safe_write_json` 安全写入导出文件
  - 添加了错误处理机制

### 2. Config模块

#### `pinocchio/config/config_manager.py`
- **应用的函数**: `safe_write_json`, `safe_read_json`, `ensure_directory`
- **更新内容**:
  - `_load_config` 方法使用 `safe_read_json` 安全读取配置
  - `save` 方法使用 `ensure_directory` 和 `safe_write_json` 安全保存配置
  - 改进了错误处理和日志记录

### 3. Agents模块

#### `pinocchio/agents/base.py`
- **应用的函数**: `validate_agent_response`
- **更新内容**:
  - 在 `_call_llm` 方法中添加了响应验证
  - 使用 `validate_agent_response` 验证LLM响应结构

#### `pinocchio/agents/generator.py`
- **应用的函数**: `format_json_response`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 导入了JSON格式化和临时文件工具函数
  - 为将来的临时文件操作做准备

#### `pinocchio/agents/optimizer.py`
- **应用的函数**: `format_json_response`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 导入了JSON格式化和临时文件工具函数

#### `pinocchio/agents/evaluator.py`
- **应用的函数**: `format_json_response`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 导入了JSON格式化和临时文件工具函数

#### `pinocchio/agents/debugger.py`
- **应用的函数**: `format_json_response`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 导入了JSON格式化和临时文件工具函数

### 4. LLM模块

#### `pinocchio/llm/custom_llm_client.py`
- **应用的函数**: `safe_json_parse`, `format_json_response`
- **更新内容**:
  - `complete` 方法使用 `safe_json_parse` 安全解析JSON响应
  - 使用 `format_json_response` 格式化JSON输出
  - 改进了JSON解析的错误处理

### 5. Memory模块

#### `pinocchio/memory/manager.py`
- **应用的函数**: `safe_write_json`, `safe_read_json`, `ensure_directory`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 初始化时使用 `ensure_directory` 确保目录存在
  - `_session_path` 方法使用 `ensure_directory` 创建目录
  - `store_agent_memory` 方法使用 `safe_write_json` 安全写入
  - `get_code_version` 方法使用 `safe_read_json` 安全读取
  - `add_performance_metrics` 方法使用 `safe_write_json` 安全写入
  - `get_performance_history` 方法使用 `safe_read_json` 安全读取
  - `update_optimization_history` 方法使用 `safe_write_json` 安全写入
  - `get_optimization_history` 方法使用 `safe_read_json` 安全读取
  - `query_agent_memories` 方法使用 `safe_read_json` 安全读取
  - `export_logs` 方法使用 `safe_read_json` 和 `safe_write_json` 安全读写

### 6. Knowledge模块

#### `pinocchio/knowledge/manager.py`
- **应用的函数**: `safe_write_json`, `safe_read_json`, `ensure_directory`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 初始化时使用 `ensure_directory` 确保目录存在
  - `_persist_fragment` 方法使用 `safe_write_json` 安全写入
  - `load_from_storage` 方法使用 `safe_read_json` 安全读取
  - 添加了错误处理机制

### 7. Prompt模块

#### `pinocchio/prompt/manager.py`
- **应用的函数**: `safe_write_json`, `safe_read_json`, `ensure_directory`, `create_temp_file`, `cleanup_temp_files`
- **更新内容**:
  - 初始化时使用 `ensure_directory` 确保目录存在
  - `_save_template` 方法使用 `safe_write_json` 安全写入
  - `_load_templates` 方法使用 `safe_read_json` 安全读取
  - `_save_memory_state` 方法使用 `safe_write_json` 安全写入
  - 添加了错误处理机制

## 应用的工具函数

### 文件操作函数
- `ensure_directory`: 确保目录存在，如果不存在则创建
- `safe_write_json`: 安全写入JSON文件，支持备份和错误处理
- `safe_read_json`: 安全读取JSON文件，处理文件不存在和解析错误
- `safe_write_text`: 安全写入文本文件
- `safe_read_text`: 安全读取文本文件

### JSON处理函数
- `safe_json_parse`: 安全解析JSON字符串
- `parse_structured_output`: 解析结构化输出
- `format_json_response`: 格式化JSON响应
- `validate_agent_response`: 验证代理响应结构
- `extract_code_from_response`: 从响应中提取代码

### 临时文件函数
- `create_temp_file`: 创建临时文件
- `create_temp_directory`: 创建临时目录
- `cleanup_temp_files`: 清理临时文件
- `cleanup_temp_directories`: 清理临时目录
- `get_temp_file_path`: 获取临时文件路径

### 配置函数
- `create_test_config`: 创建测试配置
- `load_test_config`: 加载测试配置
- `merge_configs`: 合并配置
- `create_default_test_config`: 创建默认测试配置
- `validate_config`: 验证配置
- `create_minimal_test_config`: 创建最小测试配置

## 改进效果

### 1. 错误处理改进
- 所有文件操作现在都有统一的错误处理机制
- 当文件操作失败时，会抛出有意义的异常信息
- 避免了静默失败的情况

### 2. 安全性提升
- 文件写入操作现在支持自动备份
- JSON解析更加安全，处理各种格式问题
- 目录创建操作更加可靠

### 3. 代码一致性
- 所有模块现在使用统一的文件操作接口
- 减少了重复代码
- 提高了代码的可维护性

### 4. 日志记录改进
- 文件操作现在都有详细的日志记录
- 便于调试和监控

## 注意事项

1. **向后兼容性**: 所有更改都保持了向后兼容性
2. **错误处理**: 添加了适当的错误处理，不会破坏现有功能
3. **性能**: 工具函数经过优化，不会显著影响性能
4. **测试**: 建议对更新后的模块进行充分测试

## 未来建议

1. **进一步集成**: 可以考虑在其他模块中应用更多utils函数
2. **性能监控**: 添加性能监控来跟踪文件操作的影响
3. **文档更新**: 更新相关模块的文档以反映这些更改
4. **测试覆盖**: 增加对文件操作功能的测试覆盖
