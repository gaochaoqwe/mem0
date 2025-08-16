"""
嵌入模型基类 - 简化版，适配Python 3.8
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from lite_memory.config import EmbeddingConfig


class EmbeddingBase(ABC):
    """嵌入模型基类"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化嵌入模型
        
        Args:
            config: 嵌入配置，为None时使用默认配置
        """
        self.config = config or EmbeddingConfig()
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        将文本转换为嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            List[float]: 嵌入向量
        """
        pass
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        return [self.embed(text) for text in texts]
    
    @property
    def embedding_dims(self) -> int:
        """获取嵌入维度"""
        return self.config.embedding_dims