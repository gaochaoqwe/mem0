"""
LiteMemory - 轻量级AI记忆模块
适用于Python 3.8和Windows 7的离线AI记忆系统
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "轻量级AI记忆模块 - 支持离线部署的AI记忆系统"

# 读取版本号
def read_version():
    version_file = os.path.join("lite_memory", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="lite-memory",
    version=read_version(),
    author="LiteMemory Team",
    author_email="lite@memory.dev",
    description="轻量级AI记忆模块 - 适用于Python 3.8和Windows 7",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/lite-memory/lite-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",  # Python 3.8兼容版本
        "requests>=2.25.0",
        "ollama>=0.1.0",
        "faiss-cpu>=1.7.0",  # CPU版本，兼容性更好
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.0"],  # GPU版本（可选）
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lite-memory=lite_memory.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "ai", "memory", "embedding", "vector-database", 
        "faiss", "ollama", "offline", "lightweight",
        "python38", "windows7"
    ],
    project_urls={
        "Bug Reports": "https://github.com/lite-memory/lite-memory/issues",
        "Source": "https://github.com/lite-memory/lite-memory",
        "Documentation": "https://lite-memory.readthedocs.io/",
    },
)