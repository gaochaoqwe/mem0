"""
配置类 - 简化版配置系统
适配Python 3.8，移除复杂的pydantic依赖
"""

import os
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """嵌入模型配置类"""
    model: str = "nomic-embed-text"
    embedding_dims: int = 768
    ollama_base_url: str = "http://localhost:11434"
    
    def __post_init__(self):
        # 环境变量覆盖
        if os.getenv("OLLAMA_BASE_URL"):
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")


@dataclass 
class VectorStoreConfig:
    """向量存储配置类"""
    collection_name: str = "lite_memory"
    path: Optional[str] = None
    distance_strategy: str = "cosine"  # cosine, euclidean, inner_product
    normalize_L2: bool = True
    embedding_dims: int = 768
    
    def __post_init__(self):
        if self.path is None:
            # Windows 7兼容路径
            if os.name == 'nt':
                self.path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "LiteMemory")
            else:
                self.path = os.path.join(os.path.expanduser("~"), ".lite_memory")


@dataclass
class MemoryConfig:
    """记忆模块配置类"""
    embedding_config: EmbeddingConfig = None
    vector_store_config: VectorStoreConfig = None
    auto_save: bool = True
    max_memory_size: int = 10000  # 最大记忆条数
    
    def __post_init__(self):
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()
        if self.vector_store_config is None:
            self.vector_store_config = VectorStoreConfig()
        
        # 确保嵌入维度一致
        self.vector_store_config.embedding_dims = self.embedding_config.embedding_dims


class MemoryItem:
    """记忆项数据类 - 兼容Python 3.8"""
    
    def __init__(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None
    ):
        self.id = id
        self.text = text
        self.metadata = metadata or {}
        self.created_at = created_at
        self.updated_at = updated_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典创建记忆项"""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )