"""
Lite Memory - 轻量级AI记忆模块
适用于Python 3.8和Windows 7的离线AI记忆系统

主要功能:
- FAISS向量存储
- Ollama嵌入模型
- 轻量级记忆管理
"""

__version__ = "0.1.0"
__author__ = "Lite Memory Team"

from lite_memory.memory import LiteMemory
from lite_memory.embeddings import OllamaEmbedding
from lite_memory.vector_stores import FAISSStore

__all__ = [
    "LiteMemory",
    "OllamaEmbedding", 
    "FAISSStore",
]