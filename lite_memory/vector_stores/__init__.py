"""
向量存储模块 - 提供向量存储和检索功能
"""

from lite_memory.vector_stores.base import VectorStoreBase
from lite_memory.vector_stores.faiss_store import FAISSStore

__all__ = [
    "VectorStoreBase",
    "FAISSStore",
]