"""
向量存储基类 - 简化版，适配Python 3.8
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class SearchResult:
    """搜索结果类 - 兼容Python 3.8"""
    
    def __init__(self, id: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.score = score
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata
        }


class VectorStoreBase(ABC):
    """向量存储基类"""
    
    def __init__(self, config=None):
        """
        初始化向量存储
        
        Args:
            config: 向量存储配置
        """
        self.config = config
    
    @abstractmethod
    def create_collection(self, name: str) -> bool:
        """
        创建集合
        
        Args:
            name: 集合名称
            
        Returns:
            bool: 是否创建成功
        """
        pass
    
    @abstractmethod
    def insert(self, vectors: List[List[float]], ids: List[str], 
               metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        插入向量
        
        Args:
            vectors: 向量列表
            ids: ID列表
            metadata: 元数据列表
            
        Returns:
            bool: 是否插入成功
        """
        pass
    
    @abstractmethod
    def search(self, vector: List[float], limit: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        搜索相似向量
        
        Args:
            vector: 查询向量
            limit: 返回结果数量
            filters: 过滤条件
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """
        删除向量
        
        Args:
            id: 向量ID
            
        Returns:
            bool: 是否删除成功
        """
        pass
    
    @abstractmethod
    def update(self, id: str, vector: Optional[List[float]] = None, 
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新向量
        
        Args:
            id: 向量ID
            vector: 新向量
            metadata: 新元数据
            
        Returns:
            bool: 是否更新成功
        """
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[SearchResult]:
        """
        获取向量
        
        Args:
            id: 向量ID
            
        Returns:
            Optional[SearchResult]: 向量数据，不存在时返回None
        """
        pass
    
    @abstractmethod
    def list_all(self, filters: Optional[Dict[str, Any]] = None, 
                 limit: int = 100) -> List[SearchResult]:
        """
        列出所有向量
        
        Args:
            filters: 过滤条件
            limit: 返回结果数量
            
        Returns:
            List[SearchResult]: 向量列表
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        获取向量总数
        
        Returns:
            int: 向量总数
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        清空所有向量
        
        Returns:
            bool: 是否清空成功
        """
        pass