"""
核心记忆模块 - 轻量级AI记忆系统
重构自原项目，去除LLM依赖，保留核心记忆功能
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

from lite_memory.config import MemoryConfig, MemoryItem
from lite_memory.embeddings import OllamaEmbedding
from lite_memory.vector_stores import FAISSStore

# 设置日志
logger = logging.getLogger(__name__)


class LiteMemory:
    """轻量级AI记忆系统
    
    提供文本记忆的存储、检索和管理功能
    不依赖LLM，专注于向量相似性搜索
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化记忆系统
        
        Args:
            config: 记忆配置，为None时使用默认配置
        """
        self.config = config or MemoryConfig()
        
        # 初始化嵌入模型
        self.embedding_model = OllamaEmbedding(self.config.embedding_config)
        
        # 初始化向量存储
        self.vector_store = FAISSStore(self.config.vector_store_config)
        
        logger.info("LiteMemory初始化完成")
    
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None,
            memory_id: Optional[str] = None) -> str:
        """
        添加记忆
        
        Args:
            text: 要记忆的文本内容
            metadata: 元数据
            memory_id: 指定记忆ID，为None时自动生成
            
        Returns:
            str: 记忆ID
        """
        if not text or not text.strip():
            raise ValueError("文本内容不能为空")
        
        # 生成记忆ID
        if memory_id is None:
            memory_id = str(uuid.uuid4())
        
        # 准备元数据
        memory_metadata = metadata.copy() if metadata else {}
        memory_metadata.update({
            "text": text.strip(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        
        try:
            # 生成嵌入向量
            embedding = self.embedding_model.embed(text.strip())
            
            # 存储到向量数据库
            success = self.vector_store.insert(
                vectors=[embedding],
                ids=[memory_id],
                metadata=[memory_metadata]
            )
            
            if success:
                logger.info(f"成功添加记忆: {memory_id}")
                return memory_id
            else:
                raise RuntimeError("向量存储失败")
                
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            raise RuntimeError(f"添加记忆失败: {e}")
    
    def search(self, query: str, limit: int = 5, 
               filters: Optional[Dict[str, Any]] = None,
               min_score: Optional[float] = None) -> List[MemoryItem]:
        """
        搜索相关记忆
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            filters: 过滤条件
            min_score: 最小相似度分数阈值
            
        Returns:
            List[MemoryItem]: 相关记忆列表
        """
        if not query or not query.strip():
            return []
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.embed(query.strip())
            
            # 向量搜索
            search_results = self.vector_store.search(
                vector=query_embedding,
                limit=limit,
                filters=filters
            )
            
            # 转换为记忆项
            memories = []
            for result in search_results:
                # 应用分数阈值过滤
                if min_score is not None and result.score < min_score:
                    continue
                
                memory_item = MemoryItem(
                    id=result.id,
                    text=result.metadata.get("text", ""),
                    metadata=result.metadata,
                    created_at=result.metadata.get("created_at"),
                    updated_at=result.metadata.get("updated_at")
                )
                
                # 添加相似度分数到元数据
                memory_item.metadata["similarity_score"] = result.score
                memories.append(memory_item)
            
            logger.info(f"搜索到 {len(memories)} 条相关记忆")
            return memories
            
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """
        获取指定记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            Optional[MemoryItem]: 记忆项，不存在时返回None
        """
        try:
            result = self.vector_store.get(memory_id)
            if result is None:
                return None
            
            return MemoryItem(
                id=result.id,
                text=result.metadata.get("text", ""),
                metadata=result.metadata,
                created_at=result.metadata.get("created_at"),
                updated_at=result.metadata.get("updated_at")
            )
            
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return None
    
    def update(self, memory_id: str, text: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            text: 新文本内容
            metadata: 新元数据
            
        Returns:
            bool: 是否更新成功
        """
        try:
            # 获取现有记忆
            existing = self.vector_store.get(memory_id)
            if existing is None:
                logger.warning(f"记忆 {memory_id} 不存在")
                return False
            
            # 准备更新数据
            new_metadata = existing.metadata.copy()
            new_metadata["updated_at"] = datetime.now().isoformat()
            
            if metadata:
                new_metadata.update(metadata)
            
            # 如果更新文本内容
            if text is not None and text.strip():
                new_metadata["text"] = text.strip()
                # 重新生成嵌入向量
                new_embedding = self.embedding_model.embed(text.strip())
                success = self.vector_store.update(
                    id=memory_id,
                    vector=new_embedding,
                    metadata=new_metadata
                )
            else:
                # 只更新元数据
                success = self.vector_store.update(
                    id=memory_id,
                    metadata=new_metadata
                )
            
            if success:
                logger.info(f"成功更新记忆: {memory_id}")
            return success
            
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            return False
    
    def delete(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            success = self.vector_store.delete(memory_id)
            if success:
                logger.info(f"成功删除记忆: {memory_id}")
            return success
            
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False
    
    def list_all(self, filters: Optional[Dict[str, Any]] = None,
                 limit: int = 100) -> List[MemoryItem]:
        """
        列出所有记忆
        
        Args:
            filters: 过滤条件
            limit: 返回结果数量
            
        Returns:
            List[MemoryItem]: 记忆列表
        """
        try:
            results = self.vector_store.list_all(filters=filters, limit=limit)
            
            memories = []
            for result in results:
                memory_item = MemoryItem(
                    id=result.id,
                    text=result.metadata.get("text", ""),
                    metadata=result.metadata,
                    created_at=result.metadata.get("created_at"),
                    updated_at=result.metadata.get("updated_at")
                )
                memories.append(memory_item)
            
            return memories
            
        except Exception as e:
            logger.error(f"列出记忆失败: {e}")
            return []
    
    def clear_all(self) -> bool:
        """
        清空所有记忆
        
        Returns:
            bool: 是否清空成功
        """
        try:
            success = self.vector_store.clear()
            if success:
                logger.info("成功清空所有记忆")
            return success
            
        except Exception as e:
            logger.error(f"清空记忆失败: {e}")
            return False
    
    def count(self) -> int:
        """
        获取记忆总数
        
        Returns:
            int: 记忆总数
        """
        return self.vector_store.count()
    
    def add_batch(self, texts: List[str], 
                  metadata_list: Optional[List[Dict[str, Any]]] = None,
                  memory_ids: Optional[List[str]] = None) -> List[str]:
        """
        批量添加记忆
        
        Args:
            texts: 文本列表
            metadata_list: 元数据列表
            memory_ids: 记忆ID列表
            
        Returns:
            List[str]: 成功添加的记忆ID列表
        """
        if not texts:
            return []
        
        # 生成ID列表
        if memory_ids is None:
            memory_ids = [str(uuid.uuid4()) for _ in texts]
        elif len(memory_ids) != len(texts):
            raise ValueError("记忆ID数量与文本数量不匹配")
        
        # 准备元数据列表
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        elif len(metadata_list) != len(texts):
            raise ValueError("元数据数量与文本数量不匹配")
        
        try:
            # 批量生成嵌入向量
            embeddings = self.embedding_model.embed_batch(texts)
            
            # 准备批量元数据
            batch_metadata = []
            for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
                memory_metadata = metadata.copy()
                memory_metadata.update({
                    "text": text.strip(),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
                batch_metadata.append(memory_metadata)
            
            # 批量插入
            success = self.vector_store.insert(
                vectors=embeddings,
                ids=memory_ids,
                metadata=batch_metadata
            )
            
            if success:
                logger.info(f"成功批量添加 {len(texts)} 条记忆")
                return memory_ids
            else:
                raise RuntimeError("批量向量存储失败")
                
        except Exception as e:
            logger.error(f"批量添加记忆失败: {e}")
            return []
    
    def similar_memories(self, memory_id: str, limit: int = 5) -> List[MemoryItem]:
        """
        查找与指定记忆相似的其他记忆
        
        Args:
            memory_id: 记忆ID
            limit: 返回结果数量
            
        Returns:
            List[MemoryItem]: 相似记忆列表
        """
        # 获取指定记忆
        memory = self.get(memory_id)
        if memory is None:
            logger.warning(f"记忆 {memory_id} 不存在")
            return []
        
        # 搜索相似记忆
        similar = self.search(memory.text, limit=limit + 1)  # +1是因为会包含自己
        
        # 排除自己
        return [m for m in similar if m.id != memory_id][:limit]
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取记忆系统信息
        
        Returns:
            Dict[str, Any]: 系统信息
        """
        return {
            "memory_count": self.count(),
            "embedding_model": self.config.embedding_config.model,
            "embedding_dims": self.config.embedding_config.embedding_dims,
            "vector_store_info": self.vector_store.get_info(),
            "max_memory_size": self.config.max_memory_size
        }
    
    def save(self):
        """手动保存数据"""
        self.vector_store.save()
        logger.info("记忆数据已保存")
    
    def test_connection(self) -> bool:
        """
        测试系统连接状态
        
        Returns:
            bool: 是否连接正常
        """
        try:
            # 测试嵌入模型连接
            embedding_ok = self.embedding_model.test_connection()
            
            # 测试向量存储
            vector_store_ok = self.vector_store.count() >= 0
            
            return embedding_ok and vector_store_ok
            
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False