#!/bin/bash

# 脚本用于设置开发环境和安装pre-commit钩子

# 设置错误时退出
set -e

echo "=== 设置Pinocchio开发环境 ==="

# 检查是否安装了Poetry
if ! command -v poetry &> /dev/null; then
    echo "Poetry未安装，正在安装..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "Poetry已安装"
fi

# 安装项目依赖
echo "安装项目依赖..."
poetry install

# 安装pre-commit钩子
echo "安装pre-commit钩子..."
poetry run pre-commit install

# 运行pre-commit钩子
echo "运行pre-commit钩子初始化..."
poetry run pre-commit run --all-files || true

echo "=== 开发环境设置完成 ==="
echo "可以使用以下命令运行测试:"
echo "  poetry run pytest"
echo "可以使用以下命令格式化代码:"
echo "  poetry run black ."
echo "  poetry run isort ."
echo ""
echo "Git提交前会自动运行pre-commit钩子进行代码检查和格式化" 