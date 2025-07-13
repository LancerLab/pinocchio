# Utility模块设计文档

## 概述

Utility模块 (`@/utils`) 是Pinocchio项目的核心工具库，为各个模块提供标准化的工具函数。该模块采用模块化设计，提供文件操作、JSON解析、配置管理、临时文件处理等核心功能。

## 设计原则

### 1. 统一性
- 所有模块使用相同的工具函数接口
- 标准化的错误处理和返回值格式
- 一致的命名规范和代码风格

### 2. 安全性
- 安全的文件操作，防止数据丢失
- 异常处理和错误恢复机制
- 输入验证和类型检查

### 3. 可扩展性
- 模块化设计，易于添加新功能
- 清晰的接口定义和文档
- 向后兼容的API设计

### 4. 性能优化
- 高效的算法实现
- 内存使用优化
- 缓存机制支持

## 模块结构

```
pinocchio/utils/
├── __init__.py          # 模块入口，导出所有公共接口
├── file_utils.py        # 文件操作工具
├── json_parser.py       # JSON解析工具
├── config_utils.py      # 配置管理工具
└── temp_utils.py        # 临时文件工具
```

## 核心功能模块

### 1. 文件操作模块 (`file_utils.py`)

#### 功能概述
提供安全的文件读写、目录管理、文件信息获取等功能。

#### 核心函数

##### `safe_write_json(file_path: str, data: Any, **kwargs) -> bool`
安全写入JSON文件
```python
# 使用示例
success = safe_write_json("data.json", {"key": "value"}, indent=2)
if success:
    print("文件写入成功")
```

##### `safe_read_json(file_path: str, default: Any = None) -> Any`
安全读取JSON文件
```python
# 使用示例
data = safe_read_json("data.json", default={})
print(f"读取的数据: {data}")
```

##### `safe_write_text(file_path: str, content: str, encoding: str = "utf-8") -> bool`
安全写入文本文件
```python
# 使用示例
success = safe_write_text("output.txt", "Hello, World!")
```

##### `safe_read_text(file_path: str, encoding: str = "utf-8", default: str = "") -> str`
安全读取文本文件
```python
# 使用示例
content = safe_read_text("input.txt")
print(f"文件内容: {content}")
```

##### `ensure_directory(directory_path: str) -> bool`
确保目录存在
```python
# 使用示例
if ensure_directory("./data/sessions"):
    print("目录创建成功")
```

##### `get_unique_filename(base_path: str, extension: str = "") -> str`
生成唯一文件名
```python
# 使用示例
unique_file = get_unique_filename("./data/session", ".json")
# 输出: ./data/session_20241201_143022.json
```

##### `cleanup_old_files(directory: str, pattern: str, max_age_hours: int = 24) -> int`
清理旧文件
```python
# 使用示例
cleaned_count = cleanup_old_files("./temp", "*.tmp", max_age_hours=1)
print(f"清理了 {cleaned_count} 个旧文件")
```

##### `get_file_info(file_path: str) -> Dict[str, Any]`
获取文件信息
```python
# 使用示例
info = get_file_info("data.json")
print(f"文件大小: {info['size']} bytes")
print(f"修改时间: {info['modified']}")
```

### 2. JSON解析模块 (`json_parser.py`)

#### 功能概述
提供结构化输出解析、代码提取、响应验证等功能。

#### 核心函数

##### `safe_json_parse(text: str, default: Any = None) -> Any`
安全解析JSON字符串
```python
# 使用示例
try:
    data = safe_json_parse('{"key": "value"}')
    print(f"解析结果: {data}")
except ValueError:
    print("JSON解析失败")
```

##### `parse_structured_output(response: str) -> Dict[str, Any]`
解析结构化LLM输出
```python
# 使用示例
llm_response = '''
{
    "success": true,
    "output": {
        "code": "func test() { ... }",
        "explanation": "Generated function"
    }
}
'''
result = parse_structured_output(llm_response)
print(f"代码: {result['output']['code']}")
```

##### `extract_code_from_response(response: str, language: str = "python") -> str`
从响应中提取代码
```python
# 使用示例
response = '''
Here's the code:
```python
def hello():
    print("Hello, World!")
```
'''
code = extract_code_from_response(response, "python")
print(f"提取的代码: {code}")
```

##### `validate_agent_response(response: Dict[str, Any]) -> bool`
验证智能体响应格式
```python
# 使用示例
response = {
    "success": True,
    "output": {"code": "..."},
    "agent_type": "generator"
}
is_valid = validate_agent_response(response)
print(f"响应格式有效: {is_valid}")
```

##### `format_json_response(data: Dict[str, Any], indent: int = 2) -> str`
格式化JSON响应
```python
# 使用示例
data = {"key": "value", "nested": {"inner": "data"}}
formatted = format_json_response(data, indent=4)
print(formatted)
```

### 3. 配置管理模块 (`config_utils.py`)

#### 功能概述
提供测试配置创建、配置合并、配置验证等功能。

#### 核心函数

##### `create_test_config(**overrides) -> Dict[str, Any]`
创建测试配置
```python
# 使用示例
config = create_test_config(
    llm_provider="openai",
    debug_enabled=True
)
print(f"测试配置: {config}")
```

##### `load_test_config(config_path: str) -> Dict[str, Any]`
加载测试配置
```python
# 使用示例
config = load_test_config("test_config.json")
print(f"加载的配置: {config}")
```

##### `merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]`
合并配置
```python
# 使用示例
base = {"app": {"name": "test"}, "llm": {"provider": "openai"}}
override = {"llm": {"model": "gpt-4"}}
merged = merge_configs(base, override)
print(f"合并后的配置: {merged}")
```

##### `validate_config(config: Dict[str, Any]) -> bool`
验证配置格式
```python
# 使用示例
config = {"app": {"name": "test"}, "llm": {"provider": "openai"}}
is_valid = validate_config(config)
print(f"配置有效: {is_valid}")
```

##### `create_default_test_config() -> Dict[str, Any]`
创建默认测试配置
```python
# 使用示例
default_config = create_default_test_config()
print(f"默认配置: {default_config}")
```

##### `create_minimal_test_config() -> Dict[str, Any]`
创建最小测试配置
```python
# 使用示例
minimal_config = create_minimal_test_config()
print(f"最小配置: {minimal_config}")
```

### 4. 临时文件模块 (`temp_utils.py`)

#### 功能概述
提供临时文件/目录创建、清理、路径管理等功能。

#### 核心函数

##### `create_temp_file(prefix: str = "temp_", suffix: str = "", directory: str = None) -> str`
创建临时文件
```python
# 使用示例
temp_file = create_temp_file("test_", ".txt")
print(f"临时文件: {temp_file}")
```

##### `create_temp_directory(prefix: str = "temp_", directory: str = None) -> str`
创建临时目录
```python
# 使用示例
temp_dir = create_temp_directory("test_")
print(f"临时目录: {temp_dir}")
```

##### `get_temp_file_path(prefix: str = "temp_", suffix: str = "") -> str`
获取临时文件路径
```python
# 使用示例
temp_path = get_temp_file_path("session_", ".json")
print(f"临时文件路径: {temp_path}")
```

##### `cleanup_temp_files(pattern: str = "temp_*") -> int`
清理临时文件
```python
# 使用示例
cleaned_count = cleanup_temp_files("test_*")
print(f"清理了 {cleaned_count} 个临时文件")
```

##### `cleanup_temp_directories(pattern: str = "temp_*") -> int`
清理临时目录
```python
# 使用示例
cleaned_count = cleanup_temp_directories("test_*")
print(f"清理了 {cleaned_count} 个临时目录")
```

## 使用模式

### 1. 模块导入模式
```python
# 推荐：从模块导入特定函数
from pinocchio.utils import safe_write_json, safe_read_json
from pinocchio.utils import parse_structured_output, extract_code_from_response

# 使用函数
safe_write_json("data.json", {"key": "value"})
data = safe_read_json("data.json")
```

### 2. 命名空间导入模式
```python
# 可选：导入整个模块
import pinocchio.utils as utils

# 使用函数
utils.safe_write_json("data.json", {"key": "value"})
utils.cleanup_temp_files()
```

### 3. 上下文管理模式
```python
# 临时文件管理
from pinocchio.utils import create_temp_file, cleanup_temp_files

# 创建临时文件
temp_file = create_temp_file("test_", ".json")
try:
    # 使用临时文件
    safe_write_json(temp_file, {"data": "value"})
    # 处理逻辑...
finally:
    # 清理临时文件
    cleanup_temp_files("test_*")
```

## 错误处理

### 1. 异常类型
```python
from pinocchio.utils import safe_read_json

try:
    data = safe_read_json("nonexistent.json")
except FileNotFoundError:
    print("文件不存在")
except json.JSONDecodeError:
    print("JSON格式错误")
except Exception as e:
    print(f"其他错误: {e}")
```

### 2. 默认值处理
```python
# 使用默认值
data = safe_read_json("data.json", default={})
content = safe_read_text("file.txt", default="")
```

### 3. 布尔返回值
```python
# 检查操作是否成功
if safe_write_json("data.json", {"key": "value"}):
    print("写入成功")
else:
    print("写入失败")
```

## 性能考虑

### 1. 文件操作优化
- 使用缓冲写入减少I/O操作
- 批量处理文件操作
- 异步文件操作支持

### 2. JSON解析优化
- 使用快速JSON解析器
- 缓存解析结果
- 流式JSON处理

### 3. 内存管理
- 及时清理临时文件
- 限制文件大小
- 使用生成器处理大文件

## 测试策略

### 1. 单元测试
```python
def test_safe_write_json():
    """测试安全JSON写入"""
    test_data = {"key": "value"}
    success = safe_write_json("test.json", test_data)
    assert success

    # 验证写入的数据
    data = safe_read_json("test.json")
    assert data == test_data
```

### 2. 集成测试
```python
def test_file_workflow():
    """测试完整文件工作流"""
    # 创建临时文件
    temp_file = create_temp_file("test_", ".json")

    # 写入数据
    test_data = {"test": "data"}
    safe_write_json(temp_file, test_data)

    # 读取数据
    data = safe_read_json(temp_file)
    assert data == test_data

    # 清理
    cleanup_temp_files("test_*")
```

### 3. 错误测试
```python
def test_error_handling():
    """测试错误处理"""
    # 测试不存在的文件
    data = safe_read_json("nonexistent.json", default={})
    assert data == {}

    # 测试无效JSON
    with open("invalid.json", "w") as f:
        f.write("invalid json")

    data = safe_read_json("invalid.json", default=None)
    assert data is None
```

## 扩展指南

### 1. 添加新功能
```python
# 在相应的模块文件中添加新函数
def new_utility_function(param: str) -> bool:
    """新工具函数的文档字符串"""
    try:
        # 实现逻辑
        return True
    except Exception:
        return False

# 在__init__.py中导出
__all__.append("new_utility_function")
```

### 2. 添加新模块
```python
# 创建新模块文件
# pinocchio/utils/new_module.py

def new_module_function():
    """新模块函数"""
    pass

# 在__init__.py中导入和导出
from .new_module import new_module_function
__all__.append("new_module_function")
```

### 3. 版本兼容性
```python
# 使用版本检查确保兼容性
import sys

if sys.version_info >= (3, 9):
    # 使用新特性
    pass
else:
    # 使用兼容实现
    pass
```

## 最佳实践

### 1. 函数设计
- 单一职责原则
- 清晰的参数和返回值
- 完整的文档字符串
- 类型注解

### 2. 错误处理
- 使用具体的异常类型
- 提供有意义的错误信息
- 实现优雅的降级策略

### 3. 性能优化
- 避免重复计算
- 使用适当的数据结构
- 实现缓存机制

### 4. 测试覆盖
- 单元测试覆盖率 > 90%
- 包含边界条件测试
- 错误场景测试

## 总结

Utility模块是Pinocchio项目的核心基础设施，提供了标准化、安全、高效的工具函数。通过模块化设计和清晰的接口，为整个项目提供了可靠的基础支持。

该模块的设计遵循了统一性、安全性、可扩展性和性能优化的原则，确保了代码的可维护性和可扩展性。通过详细的文档和测试策略，确保了模块的稳定性和可靠性。
