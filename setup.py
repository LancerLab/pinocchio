#!/usr/bin/env python3
"""Setup script for Pinocchio."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pinocchio-cli",
    version="0.1.0",
    author="Pinocchio Team",
    author_email="your.email@example.com",
    description="A multi-agent system for writing, debugging, and optimizing Choreo compute kernel DSL operators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/pinocchio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "rich>=13.5.0",
        "prompt-toolkit>=3.0.0",
        "click>=8.0.0",
        "pygments>=2.15.0",
        "pydantic>=2.5.2",
        "python-json-logger>=2.0.7",
        "pyyaml>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "pinocchio=pinocchio.cli.main:run",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
