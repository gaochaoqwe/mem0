"""
嵌入模块 - 提供文本嵌入功能
"""

from lite_memory.embeddings.base import EmbeddingBase
from lite_memory.embeddings.ollama import OllamaEmbedding

__all__ = [
    "EmbeddingBase",
    "OllamaEmbedding",
]